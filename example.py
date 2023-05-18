# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
"""
Example of how to use `taskmaster`.
"""
from random import random
from time import sleep

from mpi4py import MPI

from taskmaster import work_delegation

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def func(x):
    """Example function to process a task."""
    print(f"Sleeping on task `{x}`.")
    sleep(3 * random())


if __name__ == "__main__":
    tasks = list(range(100))
    work_delegation(func, tasks, comm, master_verbose=True,
                    worker_verbose=True)

    comm.Barrier()
    if rank == 0:
        print("Everything completed.")
