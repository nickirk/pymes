# PyMES
(**Py**thon package for **M**any-**E**lectron **S**imulations (PyMES))

PyMES package implements many-electron methods for ground state and excited states.
- Ground state:
  - MP2
  - CCD, DCD
  - CCSD, DCSD
- Excited states:
  - EOM-CCSD
  - FEAST-EOM-CCSD
  - CIF-based real-time EOM-CCSD

Key features:
  1. Handling non-hermitian Hamiltonians, such as transcorrelated Hamiltonian;
  2. Treating simple model system like 3D uniform electron gas, 
     as well as interface to popular electronic method packages via FCIDUMP.

The package is currently under active development, so some 
functions and classes may change during this time. If you experience
any problems, feel free to open issues. Please include detailed
explanations to the issues you experience when you do so.


# Installation

## Dependancies
- a modified [PySCF](https://github.com/nickirk/pyscf) for FEAST
- numpy
- scipy
- h5py
- spglib
- pytest (test purpose, not needed for running the code)

```pip install .```
should install the package.

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

Energy-filtered excited states and real-time dynamics served in a contour integral, Ke Liao, 
https://arxiv.org/abs/2409.07354

# Contributors
Ke Liao, Thomas Schraivogel, Evelin Christlmaier, Daniel Kats
