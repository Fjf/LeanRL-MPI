import logging
import time

import numpy as np
import torch
from mpi4py import MPI

from leanrl.mpi.env_wrappers import MPIFunctionParams
from leanrl.mpi.utils import make_env, Args, Profiler


def unstack_args_kwargs(args, kwargs, indices: slice):
    args = tuple(arg if type(arg) != torch.Tensor else arg[indices] for arg in args)
    kwargs = {k: v if type(v) != torch.Tensor else v[indices] for k, v in kwargs.items()}
    return args, kwargs


def main_worker(args: Args):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    env_rank = rank - 1

    profiler = Profiler()
    running = True

    while running:
        env = make_env(args.env_id, env_rank, args.capture_video, args.run_name, args.gamma)()
        method_lookup = {name: number + 1 for number, name in zip(range(len(dir(env))), dir(env))}

        do_tree_comm = args.num_comm_tree_chunks != -1
        is_branch = False
        chunk_size = 1
        my_step_source = 0
        if do_tree_comm:
            chunk_size = (comm.Get_size() - 1) // args.num_comm_tree_chunks
            is_branch = env_rank % chunk_size == 0

        if is_branch:
            # This worker needs to pass on data and will receive larger chunks
            action_buffer = np.zeros((chunk_size, *env.action_space.shape))
            rtt_buffer = np.zeros((chunk_size, 3), dtype=np.float32)
            observation_buffer = np.zeros((chunk_size, *env.observation_space.shape))
            my_step_source = 0
        else:
            action_buffer = np.zeros(env.action_space.shape)
            rtt_buffer = np.zeros(3, dtype=np.float32)
            observation_buffer = None
            if do_tree_comm:
                my_step_source = ((env_rank // chunk_size) * chunk_size) + 1


        while True:
            profiler.start("idle")
            while not comm.Iprobe():
                time.sleep(1e-5)

            if comm.Iprobe(tag=method_lookup["step"]):
                profiler.start("step")

                comm.Recv([action_buffer, MPI.FLOAT], tag=method_lookup["step"])

                if is_branch:
                    for i in range(1, chunk_size):
                        comm.Send([action_buffer[i], MPI.FLOAT], dest=rank + i, tag=method_lookup["step"])

                    observation, reward, terminated, truncated, info = env.step(action_buffer[0])

                    rtt_buffer[0].data[0] = reward
                    rtt_buffer[0].data[1] = terminated
                    rtt_buffer[0].data[2] = truncated
                    observation_buffer[0] = observation
                    infos = [info]
                    for i in range(1, chunk_size):
                        comm.Recv([observation_buffer[i], MPI.FLOAT], source=rank + i)
                        comm.Recv([rtt_buffer[i], MPI.FLOAT], source=rank + i)

                        info = comm.recv(source=rank + i)
                        infos.append(info)

                    comm.Send([observation_buffer, MPI.FLOAT], dest=0)
                    comm.Send([rtt_buffer, MPI.FLOAT], dest=0)
                    comm.send(info, dest=0)
                else:
                    observation, reward, terminated, truncated, info = env.step(action_buffer)
                    rtt_buffer.data[0] = reward
                    rtt_buffer.data[1] = terminated
                    rtt_buffer.data[2] = truncated

                    comm.Send([observation, MPI.FLOAT], dest=my_step_source)
                    comm.Send([rtt_buffer, MPI.FLOAT], dest=my_step_source)
                    comm.send(info, dest=my_step_source)

            else:
                params = MPIFunctionParams(*comm.recv())

                if params.fn == "close":
                    break
                if params.fn == "kill":
                    running = False
                    break

                profiler.start(params.fn)

                fn_args, fn_kwargs = unstack_args_kwargs(params.args, params.kwargs, slice(0, 1))

                rpc_func = env.__getattribute__(params.fn)
                response = rpc_func(*fn_args, **fn_kwargs)

                comm.send(response, params.source)

        # profiler.print()
