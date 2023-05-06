# Copyright (C) 2022 Richard Stiskalek
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 2 of the License, or (at your
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

from setuptools import (setup, find_packages)

with open('README.md', "r") as fh:
    long_description = fh.read()


setup(
    name='taskmaster',
    version="0.1.0",
    description='A simple Python MPI taskmaster.',
    long_description=long_description,
    url='https://github.com/Richard-Sti/TaskmasterPython',
    author='Richard Stiskalek',
    author_email='richard.stiskalek@protonmail.com',
    license='GPL-3.0',
    packages=find_packages(),
    install_requires=["mpi4py>=3.0.0"],
    python_requires=">=3.6",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
        'Intended Audience :: Science/Research',
        'Natural Language :: English'],
)
