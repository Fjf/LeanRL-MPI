import datetime
import time
from collections import defaultdict
from typing import Optional

import numpy as np
import torch
from mpi4py import MPI

from leanrl.mpi.env_wrappers import MPIFunctionParams
from leanrl.mpi.utils import make_env, Args


class Profiler:
    def __init__(self):
        self.segment_starts: dict[str, datetime] = {}
        self.segments: dict[str, float] = defaultdict(float)
        self.active_segment: Optional[str] = None

    def start(self, segment_name: str):
        if self.active_segment is not None:
            self.stop(self.active_segment)

        self.segment_starts[segment_name] = datetime.datetime.now()
        self.active_segment = segment_name

    def stop(self, segment_name):
        if segment_name not in self.segment_starts:
            return

        start_time = self.segment_starts[segment_name]
        end_time = datetime.datetime.now()
        duration = (end_time - start_time).total_seconds()
        self.segments[segment_name] += duration

        del self.segment_starts[segment_name]
        if self.active_segment == segment_name:
            self.active_segment = None

    def get_segment_time(self, segment_name):
        return self.segments.get(segment_name, 0.0)

    def print(self):
        for segment, time in self.segments.items():
            print(f"{segment}: {time} seconds")


def unstack_args_kwargs(args, kwargs, indices: slice):
    args = tuple(arg if type(arg) != torch.Tensor else arg[indices] for arg in args)
    kwargs = {k: v if type(v) != torch.Tensor else v[indices] for k, v in kwargs.items()}
    return args, kwargs


def main_worker(args: Args):
    comm = MPI.COMM_WORLD

    profiler = Profiler()
    running = True

    while running:
        env = make_env(args.env_id, comm.Get_rank() - 1, args.capture_video, args.run_name, args.gamma)()
        method_lookup = {name: number + 1 for number, name in zip(range(len(dir(env))), dir(env))}
        action_buffer = np.zeros(env.action_space.shape)
        while True:
            profiler.start("idle")
            while not comm.Iprobe():
                time.sleep(1e-5)

            if comm.Iprobe(tag=method_lookup["step"]):
                profiler.start("step")
                comm.Recv(action_buffer.data, tag=method_lookup["step"])
                response = env.step(action_buffer)
                
                comm.Send(response[0].data, dest=0)
                comm.send(response[1:], dest=0)

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

        profiler.print()
