from mpi4py import MPI
from taskmaster import master_process, worker_process

from time import sleep
from random import random


# setup MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def func(x):
    """Example function to process a task."""
    sleep(3 * random())


# Typical work delegation if-else. rank = 0 delegates the tasks.
if rank == 0:
    tasks = list(range(100))
    master_process(tasks, comm)
else:
    worker_process(func, comm, True)

# Synchronise CPUs
comm.Barrier()

# Celebrate
if rank == 0:
    print("Everything completed.")
