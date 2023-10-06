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

PyMES can be installed by pip 

```bash
pip install ./
```

PyMES depends on the C++ Cyclops Tensor Framework (CTF) library. Therefore, 
you need to build the CTF before using pymes.

## Building CTF
**Some legacy parts of the code still relies on CTF library, which will be removed in the future, as the library is not 
constantly well maintained. Newly added modules like scf do not have this dependency.**

Building instructions can be found at https://github.com/cyclops-community/ctf.  
Because of popular demand we outline the steps and some useful tips in the following.  

First, pull the ctf lib source using

```bash
cd pymes/lib
git submodule init
git submodule update
```
Now you should see that the ctf directory contain the source files. Then
```bash
cd ctf
```

Run the configure script to check if all necessary libraries etc. can be found.
```
./configure --install-dir=/path/to/install/dir
```
**Tip:** Make sure your MPI, OpenMP and dynamic BLAS and LAPACK libraries are in your PATH.  
It might complain that static libs cannot be found, but as long as the **dynamic** ones are found
you can proceed to next steps.

**Tip:** If you use the --install-dir option, it is necessary to export 
the path to the libraries in your bash as

```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/install/dir/lib
```
**Tip:** Note the suffixed /lib.  

Now we can build ctf with
```
make -j 8
make python -j 8
```
**Tip:** It may be necessary to make some changes in the `Makefile`. 
For example changing the `python` command to `python3` on line 121 
```MakeFile
LDFLAGS="-L$(BDIR)/lib_shared" python3 setup.py build_ext -j4 --force -b $(BDIR)/lib_python/ -t $(BDIR)/lib_python/; \
```
and the `pip` to `pip3` on line 133
```MakeFile
pip3 install --force --user -b $(BDIR)/lib_python/ . --upgrade; \
```
, if cython is only available in your
`python3` or when you `make python -j 8` it
complains that it cannot find `python`.  

If that works, pip install ctf with
```
make python_install
```
**Tip:** If you don't have sudo, it may also be necessary to add to the pip install command in the Makefile the ```--user``` flag.  

Now try to import 'ctf' in python
```python3
import ctf
```
to see if any errors appear. 

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
Ke Liao, Thomas Schraivogel, Evelin Christlmaier, Daniel Kats
