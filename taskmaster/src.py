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
from copy import deepcopy
from datetime import datetime

from mpi4py import MPI


def master_process(tasks, comm, verbose=False):
    """
    Check if a process of higher rank is available, if so delegates it a task.
    This must be run from the 0th rank and will block until all tasks have been
    delegated.

    Arguments
    ---------
    tasks : list
        List of arguments to be send to the function evaluated by a worker.
    comm : `py:class:mpi4py:MPI.COMM_WORLD`
        MPI Comm world object.
    verbose : bool
        Verbosity flag.
    """
    if not comm.Get_size() > 1:
        raise ValueError("MPI size must be > 1.")
    # We check that `tasks` is a list and that it does not contain `None`.
    # These are used to terminate the task assignment.
    if not (isinstance(tasks, list)
            and all(task is not None for task in tasks)):
        raise TypeError("`tasks` must be a list and cannot contain `None`")
    status = MPI.Status()
    nworkers = comm.Get_size() - 1
    # We put the breaking condition at the front and deepcopy it to be certain
    # since we will be modifying it.
    tasks = [None] * nworkers + deepcopy(tasks)
    while len(tasks) > 0:
        # If a a message is received, i.e. a worker is available, we send it
        # a task.
        comm.recv(source=MPI.ANY_SOURCE, status=status)
        dest = status.Get_source()
        task = tasks.pop()
        comm.send(task, dest=dest)
        if verbose and task is not None:
            print(
                f"{datetime.now()}: sending task {task} to worker {dest}.",
                f"{len(tasks) - nworkers} tasks remaining.",
                flush=True,
            )


def worker_process(func, comm, verbose=False):
    """
    Call `func(task)` for each task received from the master process.

    Arguments
    ---------
    func : `py:function`
        Function to be evaluated.
    comm : `py:class:mpi4py:MPI.COMM_WORLD`
        MPI Comm world object.
    verbose : bool
        Verbosity flag.
    """
    while True:
        # We send a signal to the master process that this rank is available.
        # We then wait to receive a task and evaluate it, unless the task is
        # a breaking condition.
        comm.send(True, dest=0)
        task = comm.recv(source=0)
        if task is None:
            break

        if verbose:
            rank = comm.Get_rank()
            now = datetime.now()
            print(f"{now}: rank {rank} received task {task}.", flush=True)
        func(task)


def work_delegation(func, tasks, comm, master_verbose=True,
                    worker_verbose=False):
    """
    Worker delegation loop. If `comm.Get_size() > 1` then the master process
    is delegating tasks to workers. Otherwise, the tasks are evaluated on the
    master process.

    Parameters
    ----------
    func : `py:function`
        Function to be evaluated.
    tasks : list
        List of arguments to be send to the function evaluated by a worker.
    comm : `py:class:mpi4py:MPI.COMM_WORLD`
        MPI Comm world object.
    master_verbose : bool
        Verbosity flag for the master process.
    worker_verbose : bool
        Verbosity flag for the worker process.

    Returns
    -------
    None
    """
    if comm.Get_size() > 1:
        if comm.Get_rank() == 0:
            master_process(tasks, comm, verbose=master_verbose)
        else:
            worker_process(func, comm, verbose=worker_verbose)
    else:
        for task in tasks:
            if master_process:
                print(f"{datetime.now()}: completing task `{task}`.",
                      flush=True)
            func(task)
