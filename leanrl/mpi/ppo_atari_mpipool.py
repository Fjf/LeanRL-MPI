import logging

from mpi4py import MPI

from leanrl.mpi.master import main
from leanrl.mpi.utils import parse_args
from leanrl.mpi.worker import main_worker

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
    )

    logging.info("Parsing args")
    args = parse_args()

    logging.info("Doing MPI stuff")
    comm = MPI.COMM_WORLD
    num_processes = comm.Get_size()
    if num_processes < 2:
        logging.error("Cannot run MPIRL with less than 2 processes.")
    logging.info("In execution")

    if comm.Get_rank() == 0:
        main(args)
    else:
        main_worker(args)
