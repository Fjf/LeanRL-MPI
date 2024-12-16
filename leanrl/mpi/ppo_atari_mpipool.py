import logging
import os
from functools import partial

from mpi4py import MPI

from leanrl.mpi.master import main
from leanrl.mpi.utils import parse_args
from leanrl.mpi.worker import main_worker

def log_factory(*args, record_factory_fn=None, **kwargs):
    record = record_factory_fn(*args, **kwargs)
    record.rank = MPI.COMM_WORLD.Get_rank()
    return record

if __name__ == "__main__":
    old_factory = logging.getLogRecordFactory()
    logging.setLogRecordFactory(partial(log_factory, record_factory_fn=old_factory))
    logging.basicConfig(
        level=logging.INFO,
        format="{%(rank)-4s} %(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
    )


    # if os.environ.get("OMPI_COMM_WORLD_SIZE", None) is None:
    #     logging.warning("OMPI_COMM_WORLD_SIZE not set, please run this application with a parallel launcher (e.g., mpirun")

    comm = MPI.COMM_WORLD
    args = parse_args(comm.Get_rank())

    num_processes = comm.Get_size()
    if num_processes < 2:
        logging.error("Cannot run MPIRL with less than 2 processes.")

    if comm.Get_rank() == 0:
        main(args)
    else:
        main_worker(args)
