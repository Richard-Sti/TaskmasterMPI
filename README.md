# Python MPI Taskmaster

A simple Python MPI taskmaster that efficiently delegates the evaluation of a function ``func(task)`` to multiple worker processes. The function should not take any additional arguments or return any values. Instead, it should write its output directly to disk.

## Overview
The taskmaster is designed to work with MPI, where the zeroth rank process acts as the taskmaster, while higher rank processes serve as workers. The taskmaster sends tasks to available workers and ensures optimal utilization of resources.

## Example
```python
from time import sleep
from mpi4py import MPI
from taskmaster import work_delegation

comm = MPI.COMM_WORLD


def func(x):
    print(f"Sleeping... Task is `{x}`.")
    sleep(3)


tasks = list(range(100))
work_delegation(func, tasks, comm)
```

## Installation
Clone the repository and install locally in your favourite environment:

```bashrc
git clone git@github.com:Richard-Sti/TaskmasterMPI.git
pip install ./TaskmasterMPI/.
```


## License
[GPL-3.0](https://www.gnu.org/licenses/gpl-3.0.en.html)
