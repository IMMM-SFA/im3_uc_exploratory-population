# sa_popgrid
Code and process for conducting sensitivity analysis for the gridded population gravity model on a cluster

## Getting Started using `sa_popgrid` on a cluster

### Note
The following code was tested on THECUBE cluster courtesy of the Reed Research Group at Cornell University.  This cluster has 32 compute nodes with dual 8-core Intel E52680 CPUs @ 2.7 GHz with 128 GB of RAM running OpenHPC v1.3.8 with CentOS 7.6.

### STEP 1:  Installing GDAL
This code has a GDAL 2.2.3 dependency.  This was installed using the following:

```sh
# download GDAL
wget http://download.osgeo.org/gdal/2.2.3/gdal-2.2.3.tar.gz

# untar
tar xzf gdal-2.2.3.tar.gz
cd gdal-2.2.3

# compile from source
./configure --with-libkml
make

# install to a local path...I chose my home dir and the following commands assume that
export DESTDIR="$HOME" && make -j4 install

# add bin to path; if you do not have a bash_profile setup run...
vim ~/.bash_profile

# and add in the following or append to an existing PATH variable and libs variable
PATH=$PATH:$HOME/usr/local/bin
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/usr/local/lib

# save and exit vim and source your changes using
source ~/.bash_profile
```


