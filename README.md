# TaskmasterMPI


A simple Python taskmaster that delagates the evaluation of a function ``f(task)`` where ``task`` is an element of a vector tasks. A new task is delegated whenever a worker is free. Function ``f(x)`` is not expected to take any other arguments and or return anything, instead it should write its output directly to disk and the user can then read those in.


## Installation
Clone the repository and install locally:

```bashrc
git clone git@github.com:Richard-Sti/TaskmasterMPI.git
pip install ./TaskmasterMPI/.
```


## License
[GPL-3.0](https://www.gnu.org/licenses/gpl-3.0.en.html)