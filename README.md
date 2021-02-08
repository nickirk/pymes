# PyMeS

**Py**thon package for **M**any-**e**lectron **S**imulations (PyMeS)

# Building

PyMeS does not need to be build, just add the following line to your .bashrc or .zshrc file

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


But PyMeS depends on the Cyclops Tensor Framework (CTF). Therefore you need to build the CTF before using pymes.

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
**Tip:** Make sure your MPI, OpenMP and dynamic BLAS and LAPACK libraries are in your PATH and dont worry if the configure script is not finding the static BLAS and LAPACK libraries.  
**Tip:** If you use the --install-dir option (because you dont have sudo) it may be necessary to export the path to the libraries in your bash as
```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/install/dir/lib
```
**Tip:** Note the suffixed /lib.  

Now we can build ctf with
```
make python -j [as many cores as you want to dedicate]
```
**Tip:** It may be necessary to make some changes in the /path/to/ctf/source/code/Makefile. For example changing the python command to python3, if cython is only available in your python3.  

Test your CTF installation with
```
make python_test
```

If that works, pip install ctf with
```
make python_install
```
**Tip:** If you dont have sudo, it may also be necessary to add to the pip install command in the Makefile the ```--user``` flag.  
**Tip:** It also may be that pip somehow messes the ctf libs up while copying. Then copy and overwrite yourself with
```
cp /path/to/your/build/directory/ctf/lib_python/ctf/*.so /home/$USER/.local/lib/python3.6/site-packages/ctf/.
```



# For developers

## Test

For each functionality added, a test file or an example file on how to use it should be added under the *test* directory. For now, the test files serve the purpose of showing examples. In the future, they should also serve as unit tests, where standard and correct results should be supplied and be compared to each running results.

## Naming style

We follow the convention explained in wikipedia

> Python and Ruby
>
> [Python](https://en.wikipedia.org/wiki/Python_(programming_language)) and [Ruby](https://en.wikipedia.org/wiki/Ruby_(programming_language)) both recommend `UpperCamelCase` for class names, `CAPITALIZED_WITH_UNDERSCORES` for constants, and `lowercase_separated_by_underscores` for other names.
>
> In Python, if a name is intended to be "[private](https://en.wikipedia.org/wiki/Private_member)", it is prefixed by an underscore. Private variables are enforced in Python only by convention. Names can also be suffixed with an underscore to prevent conflict with Python keywords. Prefixing with double underscores changes behaviour in classes with regard to [name mangling](https://en.wikipedia.org/wiki/Name_mangling#Python). Prefixing *and* suffixing with double underscores are reserved for "magic names" which fulfill special behaviour in Python objects.[[39\]](https://en.wikipedia.org/wiki/Naming_convention_(programming)#cite_note-pep8-39)

Some old codes need to be refactored to fulfil the naming conventions. But newly added code should follow these conventions. 

