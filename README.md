# pymes

python package for many-electron simulations

# Building

Pymes does not need any kind of installation as of now. But you have to change the hard-coded ```sys.path.append()``` path in the input files.  
But Pymes depends on the Cyclops Tensor Framework (CTF). Therefore you need to build the CTF before using pymes.

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
./path/to/ctf/source/code/configure --install-dir=/directory/where/lib/files/will/be/installed/if/you/dont/have/sudo
```
**Tip:** Make sure your MPI, OpenMP and dynamic BLAS and LAPACK libraries are in your PATH and dont worry if the configure script is not finding the static BLAS and LAPACK libraries.  
**Tip:** If you use the --install-dir option it may be necessary to export the path to the libraries in your bash as
```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/directory/where/lib/files/will/be/installed/if/you/dont/have/sudo
```
Now we can build ctf with
```
make python -j [as many cores as you want to dedicate]
```
**Tip:** It may be necessary to make some changes in the /path/to/ctf/source/code/Makefile. For example changing the python command to python3, if cython is only available in your python3.  

To pip install ctf use
```
make python_install
```
**Tip:** If you dont have sudo, it may also be necessary to add to the pip install command in the Makefile the ```--user``` flag.  
**Tip:** It also may be that pip somehow messes the ctf libs up while copying. Then copy and overwrite yourself with
```
cp -f /path/to/your/build/directory/ctf/lib_python/ctf/* /home/$USER/.local/lib/python3.6/site-packages/ctf/.
```
Test your CTF installation with
```
make python_test
```
