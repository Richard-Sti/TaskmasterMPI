# Copyright (C) 2022  Richard Stiskalek
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
from mpi4py import MPI


def master_process(tasks, comm, verbose=False):
    """
    The master, delagating process of rank 0. Checks if processes of higher
    rank are available and if so sends some work their way.

    Arguments
    ---------
    tasks: list
        List of arguments to be send to the function evaluated by a worker.
    commm: `py:class:mpi4py:MPI.COMM_WORLD`
        MPI Comm world object.
    verbose: bool
        Verbosity flag.

    Returns
    -------
    None
    """
    # Ensure that the comm size is more than 1
    if not comm.Get_size() > 1:
        raise ValueError("MPI size must be > 1.")
    # Check we have a list
    if not isinstance(tasks, list):
        raise TypeError("`tasks` must be a list.")
    # Check the list does not cointan None
    if any(task is None for task in tasks):
        raise TypeError(tasks, "`tasks` cannot contain `None`.")
    # Put the breaking conditions at the front
    tasks = [None] * (comm.Get_size() - 1) + tasks

    status = MPI.Status()
    while len(tasks) > 0:
        # If a message receieved means more work to be delegated to the source
        comm.recv(source=MPI.ANY_SOURCE, status=status)
        dest = status.Get_source()
        # Send a task to the worker
        task = tasks.pop()
        comm.send(task, dest=dest)
        if verbose:
            print("Sending task {} to worker {}.".format(task, dest))


def worker_process(func, comm, verbose=False):
    """
    The worker process of 1 and higher that evaluates `func(task)`.

    Arguments
    ---------
    func: `py:function`
        Function to be evaluated.
    commm: `py:class:mpi4py:MPI.COMM_WORLD`
        MPI Comm world object.
    verbose: bool
        Verbosity flag.

    Returns
    -------
    None
    """
    while True:
        # Send a signal that worker can work
        comm.send(True, dest=0)
        # Receive a task
        task = comm.recv(source=0)
        # Breaking condition
        if task is None:
            break
        
        if verbose:
            print("Rank {} received task {}.".format(comm.Get_rank(), task))
        # Actually evaluate the function
        func(task)
