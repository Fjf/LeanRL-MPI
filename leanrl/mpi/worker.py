import logging
import math
import time

from mpi4py import MPI

from leanrl.mpi.utils import make_env, Args


def main_worker(args: Args):
    logging.info("Worker creating env...")
    comm = MPI.COMM_WORLD
    env = make_env(args.env_id, comm.Get_rank() - 1, args.capture_video, args.run_name, args.gamma)

    logging.info("Worker in loop...")
    while True:
        while not comm.Iprobe():
            time.sleep(1e-5)
        source, cmd, args_arr, kwargs_arr = comm.recv()

        if len(args_arr) > 1:
            """
            Code for branch worker, branch workers send chunks of the data further down the tree.
            """

            # Send N chunks of size ceil(L/N) to the first worker of every chunk
            chunk_size = math.ceil((len(args_arr) - 1) / args.num_comm_tree_chunks)
            dest_worker_ids = range(0, len(args_arr), chunk_size)
            for lo in dest_worker_ids:
                hi = min(lo + chunk_size, len(args_arr))

                comm.Isend((comm.Get_rank(), cmd, args_arr[lo:hi], kwargs_arr[lo:hi]), dest=lo)

            fn_args = args_arr[0]
            fn_kwargs = kwargs_arr[0]

            rpc_func = env.__getattribute__(cmd)
            response = rpc_func(*fn_args, **fn_kwargs)

            for worker in dest_worker_ids:
                comm.recv(source=worker)
        else:
            """
            Code for leaf worker, leaf workers do not need to send data further.
            """

            fn_args = args_arr[0]
            fn_kwargs = kwargs_arr[0]

            rpc_func = env.__getattribute__(cmd)
            response = rpc_func(*fn_args, **fn_kwargs)

        comm.send(response, source)

        if cmd == "close":
            break
