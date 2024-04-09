# PyMES
(**Py**thon package for **M**any-**E**lectron **S**imulations (PyMES))

PyMES package implements many-electron methods, such as MP2, 
CCD/DCD, CCSD/DCSD for ground states, and EOM-CCSD for excited states, using
the CTF library for automatic tensor contractions. 

Key features:
  1. Handling non-hermitian Hamiltonians, such as transcorrelated Hamiltonian;
  2. MPI and OMP parallelisation thanks to CTF;
  3. Treating simple model system like 3D uniform electron gas, 
     as well as interface to popular electronic method packages via FCIDUMP.

The package is currently under active development, so some 
functions and classes may change during this time. If you experience
any problems, feel free to open issues. Please include detailed
explanations to the issues you experience when you do so.


# Installation

## Dependancies
- numpy
- scipy
- h5py
- Cython
- CTF
- spglib
- pytest (test purpose, not needed for running the code)

## Building

PyMES does not need to be build, just add the following line to your .bashrc or .zshrc file

```bash
export PYTHONPATH=/the/directory/containing/pymes:$PYTHONPATH
```
For example, if your pymes is stored at "~/scripts/pymes", then you do the following:
```bash
export PYTHONPATH=~/scripts:$PYTHONPATH
```

and source your ~/.bashrc or ~/.zshrc file.

```bash
source ~/.bashrc
```


But PyMES depends on the Cyclops Tensor Framework (CTF). Therefore, 
you need to build the CTF before using pymes.

## Building CTF
Building instructions can be found at https://github.com/cyclops-community/ctf.  
Because of popular demand we outline the steps and some useful tips in the following.  
First clone the git repo and create a separate build directory.
```
git clone https://github.com/cyclops-community/ctf
mkdir /path/to/your/build/directory/ctf
cd /path/to/your/build/directory/ctf
```
Run the configure script to check if all necessary libraries etc. can be found.
```
./path/to/ctf/source/code/configure --install-dir=/path/to/install/dir
```
**Tip:** Make sure your MPI, OpenMP and dynamic BLAS and LAPACK libraries are in your PATH.  
**Tip:** If you use the --install-dir option, it is necessary to export 
the path to the libraries in your bash as

```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/install/dir/lib
```
**Tip:** Note the suffixed /lib.  

Now we can build ctf with
```
make python -j [as many cores as you want to dedicate]
```
**Tip:** It may be necessary to make some changes in the /path/to/ctf/source/code/Makefile. 
For example changing the python command to python3, if cython is only available in your python3.  

Test your CTF installation with
```
make python_test
```

If that works, pip install ctf with
```
make python_install
```
**Tip:** If you don't have sudo, it may also be necessary to add to the pip install command in the Makefile the ```--user``` flag.  

## Running
For example, run test_ccsd.py with 5 processes under test_ccsd with
```
mpirun -np 5 python3 test_ccsd.py
```

# For developers
Contributions are welcome. But to keep the code in the long-term
organized and readable, some simple rules should be followed.

## Testing

For each functionality added, a test file or an example file on how to use it should be added under the *test* directory. 
For now, the test files serve the purpose of showing examples. In the future, they should also serve as unit tests, 
where standard and correct results should be supplied and be compared to each running results.

## Naming style

We follow the convention explained in wikipedia

> Python and Ruby
>
> [Python](https://en.wikipedia.org/wiki/Python_(programming_language)) and [Ruby](https://en.wikipedia.org/wiki/Ruby_(programming_language)) 
> both recommend `UpperCamelCase` for class names, 
> `CAPITALIZED_WITH_UNDERSCORES` for constants, and 
> `lowercase_separated_by_underscores` for other names.
>
> In Python, if a name is intended to be "[private](https://en.wikipedia.org/wiki/Private_member)", 
> it is prefixed by an underscore. Private variables are enforced in Python only by convention. 
> Names can also be suffixed with an underscore to prevent conflict with Python keywords. 
> Prefixing with double underscores changes behaviour in classes with 
> regard to [name mangling](https://en.wikipedia.org/wiki/Name_mangling#Python). 
> Prefixing *and* suffixing with double underscores are reserved for "magic names" which fulfill special 
> behaviour in Python objects.[[39\]](https://en.wikipedia.org/wiki/Naming_convention_(programming)#cite_note-pep8-39)

Some old codes need to be refactored to fulfil the naming conventions. 
But newly added code should follow these conventions. 

**Long and self-explanatory** names are favored over short and clean ones.

## Documentation
For each function or class added, please add documentation following the most detailed 
ones you can find in the existing code as the example.

This should include a brief introduction to the function or class,
and the main attributes in a class, and the arguments and returns of
the function.

# Reference
The following paper(s) (more to be added) should be cited in your publications if
you use PyMES:

Towards efficient and accurate ab initio solutions to periodic systems via transcorrelation and coupled cluster theory,
Ke Liao, Thomas Schraivogel, Hongjun Luo, Daniel Kats, Ali Alavi,
Physical Review Research **3** 033072 (2021) https://doi.org/10.1103/PhysRevResearch.3.033072

Density Matrix Renormalization Group for Transcorrelated Hamiltonians: Ground and Excited States in Molecules,
Ke Liao, Huanchen Zhai, Evelin Martine Christlmaier, Thomas Schraivogel, Pablo López Ríos, Daniel Kats, Ali Alavi,
Journal of Chemical Theory and Computation (2023)
https://doi.org/10.1021/acs.jctc.2c01207

# Contributors
Ke Liao, Thomas Schraivogel, Evelin Christlmaier
