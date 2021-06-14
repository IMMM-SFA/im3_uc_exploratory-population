# Experiement meta-repository for population dynamics sensitivity analysis
Code and process for conducting sensitivity analysis for the gridded population gravity model on a cluster

## Getting started with this experiment on a cluster

### Note
The following code was tested on THECUBE cluster courtesy of the Reed Research Group at Cornell University.  This cluster has 32 compute nodes with dual 8-core Intel E52680 CPUs @ 2.7 GHz with 128 GB of RAM running OpenHPC v1.3.8 with CentOS 7.6.

### STEP 1:  Installing GDAL
This code has a GDAL 2.2.3 dependency.  This was installed using the following using the default compiler `gcc 8.3.0`:

```shell script
# change directories into your libs dir; make one if it does not exist
cd ~/libs

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

Running `make` will take a while.  Once you have finished the process you can test the GDAL install by executing `gdalinfo` which should return:
```
Usage: gdalinfo [--help-general] [-json] [-mm] [-stats] [-hist] [-nogcp] [-nomd]
                [-norat] [-noct] [-nofl] [-checksum] [-proj4]
                [-listmdd] [-mdd domain|`all`]*
                [-sd subdataset] [-oo NAME=VALUE]* datasetname
FAILURE: No datasource specified.
```

### STEP 2:  Create a Python Virtual Environment
We want to use Python 3.6 so execute in your home directory:

```shell script
python3 -m venv pyenv
```

In this case `pyenv` is simply the name I have chosen for my virutal environemnt.

### STEP 3:  Install the `population_gravity` and `sa_popgrid` packages and the required Python modules in your virtual environment
1.  Activate your Python virtual environment by running:
```bash
source pyenv/bin/activate
```

2. Install the `population_gravity` Python package from GitHub:
```bash
python -m pip install -e git://github.com/IMMM-SFA/population_gravity.git@main#egg=population_gravity

```


3. Install the `sa_popgrid` Python package from GitHub:
```bash
python -m pip install -e git://github.com/IMMM-SFA/sa_popgrid.git@main#egg=sa_popgrid
```

4. Confirm that the packages installed correctly by first entering a Python prompt:
```bash
python
```
and then executing:
```python
import population_gravity
import sa_popgrid
```
If no errors return then all is well.  Exit the Python prompt by excuting:
```python
exit()
```

You can use the following to generate information on a function within the package after install:

```python
import sa_popgrid

help(sa_popgrid.reproduce_experiment)
```

which will render:

```
reproduce_experiment(data_dir, simulation_output_dir, base_year, projection_year, scenario, let_fail=False)
    Reproduce the experiment for a given year using the data archived from the following publication and run
    assertion tests to ensure outputs match to a relative and absolute threshold of 1e-6; the output assertion
    tests are written for each state in the 'assertion.log' file:

    Zoraghein, H., & O’Neill, B. C. (2020). US State-level Projections of the Spatial Distribution of Population
    Consistent with Shared Socioeconomic Pathways. Sustainability, 12(8), 3374. https://doi.org/10.3390/su12083374

    This data can be downloaded from here:

    Zoraghein, H., & O'Neill, B. (2020). Data Supplement: U.S. state-level projections of the spatial distribution of
    population consistent with Shared Socioeconomic Pathways. (Version v0.1.0) [Data set].
    Zenodo. http://doi.org/10.5281/zenodo.3756179

    :param data_dir:                            Full path to the directory of the downloaded data from the publication
    :type data_dir:                             str

    :param simulation_output_dir:               Full path to the directory where you want to write the simulation results
    :type simulation_output_dir:                str

    :param base_year:                           Base year for the simulation
    :type base_year:                            int

    :param projection_year:                     Projection year for which future population data is available for the
                                                given scenario
```

## Reproduce my experiement

### Originally published data
The population gravity model that downscaled U.S. state-level projections of population to a 1km grid over for the U.S. was originally published in the following:

>Zoraghein, H., & O’Neill, B. C. (2020). US State-level Projections of the Spatial Distribution of Population Consistent with Shared Socioeconomic Pathways. Sustainability, 12(8), 3374. https://doi.org/10.3390/su12083374

The input and output data used in this publication can be found here:

>Zoraghein, H., & O'Neill, B. (2020). Data Supplement: U.S. state-level projections of the spatial distribution of population consistent with Shared Socioeconomic Pathways. (Version v0.1.0) [Data set]. Zenodo. http://doi.org/10.5281/zenodo.3756179

State-level population projections were described in this publication:

>Jiang, L., B.C. O'Neill, H. Zoraghein, and S. Dahlke. 2020. Population scenarios for U.S. states consistent with Shared Socioeconomic Pathways. Environmental Research Letters, https://doi.org/10.1088/1748-9326/aba5b1.

The data produced in Jiang et al. (2020) can be downloaded from here:

>Jiang, L., Dahlke, S., Zoraghein, H., & O'Neill, B.C. (2020). Population scenarios for U.S. states consistent with Shared Socioeconomic Pathways (Version v0.1.0) [Data set]. Zenodo. http://doi.org/10.5281/zenodo.3956412

and the state-level model code used in that publication can be found here:

>Zoraghein, H., R. Nawrotzki, L. Jiang, and S. Dahlke (2020). IMMM-SFA/statepop: v0.1.0 (Version v0.1.0). Zenodo. http://doi.org/10.5281/zenodo.3956703

### Download the original data
Download and unzip the inputs and outputs as archived in Zoraghein and O'Neill (2020) from the following Zenodo archive:  [zoraghein-oneill_population_gravity_inputs_outputs.zip](https://zenodo.org/record/3756179/files/zoraghein-oneill_population_gravity_inputs_outputs.zip?download=1)

### Reproduce the original experiment for year 2020
Zoraghein and O'Neill (2020) used 2010 as the base historical year.  Though the population projection files include 2010 as a projected year for each SSP.  The projected 2010 should not be confused with the actual observed population data in 2010; they are different.

The following runs assumes the base year of 2010 and uses the projected data for SSP2 for year 2020, along with it's published parameters, to produce a simulated run with the goal of seeing if we can reproduce the results from the experiement.

**NOTE**:  This function does not submit a job to SLURM.  I ran this locally.

```python
import sa_popgrid

# directory of the unzipped data from Zenodo
data_dir = '<your directory>/zoraghein-oneill_population_gravity_inputs_outputs'

# directory to write the outputs to for this run
simulation_output_dir = '<your target directory>'

# use 2010 as the base year and 2020 as the first projection year
base_year = 2010
projection_year = 2020
scenario = 'SSP2'

sa_popgrid.reproduce_original_experiment(data_dir, simulation_output_dir, base_year, projection_year, scenario, let_fail=False)
```

### Extending the input data to accommodate a larger kernel distance reach
The original data was created to only include grid cells from other states when running a target state for 100 km around the border of the target state.  Since we wanted to envelope the kernel distance parameter default setting of 100 km that was used in Zoraghein and O'Neill (2020) by 50 km (50 km to 150 km), we have to rebuild the original data to support this.

**NOTE**:  This function does not submit a job to SLURM.  I ran this locally.

```python
import sa_popgrid

# directory of the unzipped data from Zenodo
data_dir = '<your directory>/zoraghein-oneill_population_gravity_inputs_outputs'

# directory that will hold the newly modified inputs.
output_dir = '<your output directory>'

# list of years as 4-digit integers to process
target_year_list = [2000, 2010]

sa_popgrid.build_new_data(data_dir, output_dir, target_year_list)
```


### Create Latin Hypercube Sample (LHS) and problem dictionary
The current LHS sample and problem dictionary that has been used for testing is stored within this package and can be accessed using:

**NOTE**:  This function does not submit a job to SLURM.  I ran this locally.


```python
import pkg_resources
import pickle

import numpy as np

# get file paths
lhs_array_file = pkg_resources.resource_filename('sa_popgrid', 'data/lhs_1000_sample.npy')
lhs_problem_dict_file = pkg_resources.resource_filename('sa_popgrid', 'data/lhs_1000_problem_dict.p')

# load files
lhs_array = np.load(lhs_array_file)

with open(lhs_problem_dict_file, "rb") as prob:
    lhs_problem_dict = pickle.load(prob)

```

These represent 1000 samples for the following bounds and parameters:

```
{'num_vars': 5,
 'names': ['alpha_urban',
  'alpha_rural',
  'beta_urban',
  'beta_rural',
  'kernel_distance_meters'],
 'bounds': [[-4, 4], [-4, 4], [-4, 4], [-4, 4], [50000, 150000]]}
 ```

The LHS array and problem dictionary can be generated and written to files using SALib like the following:

```python
import pickle

import numpy as np

from SALib.sample import latin

n_samples = 1000

# create your problem dictionary
problem = {'num_vars': 5,
             'names': ['alpha_urban',
              'alpha_rural',
              'beta_urban',
              'beta_rural',
              'kernel_distance_meters'],
             'bounds': [[-4, 4], [-4, 4], [-4, 4], [-4, 4], [50000, 150000]]}

# create your LHS array
lhs_arr = latin.sample(problem, n_samples)

# write problem dictionary as a pickled file
with open('<my path to write my file.p to>', 'wb') as prob:
    pickle.dump(problem, prob)

# save array as npy file
np.save('<my path to write my file.npy to>', lhs_arr)
```


### Validation

#### Running the validation using year 2000 as the observed data to simulate year 2010 observed data using published parameter values

Year 2000 was used as the base year to calibrate to year 2010 observed data.  We want to recreate the validation.

**NOTE**:  This function does not submit a job to SLURM.  I ran this locally.

```python
import sa_popgrid

# your directory to the newly modified inputs
data_dir = '<your directory that holds the newly modified inputs>'

simulation_output_dir = '<directory to write the outputs to>'

historical_year = 2000
projection_year = 2010

# run for all states
sa_popgrid.run_validation_allstates(historical_year, projection_year, data_dir, simulation_output_dir)

# HINT:  You can also run for a single state using the following...
sa_popgrid.run_validation(target_state='<my target state>', historical_year, projection_year, data_dir, simulation_output_dir)
```

#### Run LHS runs using the observed year of 2000 as the base historical year and 2010 as the validation year for all 1000 LHS samples.  This will assume the projected population to be what is in the 2010 observed data and NOT what is in the SSPs.  

We do this step to evaluate the influence of varying parameter values on our validation data.

**NOTE:** The following is meant to run on a cluster that utilizes the SLURM to schedule jobs.

```python
import sa_popgrid


output_script_dir = '<the path to the directory that my SLURM scripts will be written to>'

# the number of samples in your sample array
n_samples = 1000

# your directory to the newly modified inputs
data_dir = '<your directory that holds the newly modified inputs>'

# directory to write the outputs to
simulation_output_dir = '<your output dir>'

# directory to write the SLURM out log files to
slurm_out_dir = '<your output dir>'

# directory of my Python virtual environment
venv_dir = '/my/env'

# year to use as the historical reference data (e.g., observed data or previous year)
hist_yr = 2000

# validation or projection year
proj_yr = 2010

# state name to process as all lower case with underscore separators
state_name = '<my_target_state_name>'

# using validation as the scenario name
scenario = 'validation'

# see help for additional option explanations
sa_popgrid.submit_slurm_array(output_script_dir,
                               n_samples, data_dir,
                               simulation_output_dir,
                               slurm_out_dir,
                               venv_dir,
                               hist_yr,
                               proj_yr,
                               state_name,
                               scenario,
                               walltime='04:00:00',
                               submit_job=True,
                               partition='normal',
                               account_name='',
                               max_jobs=10,
                               method='validation',
                               lhs_array_file=None,
                               lhs_problem_file=None,
                               write_raster=False,
                               output_total=False,
                               write_array1d=True)
```

**NOTE:** I chose to write the outputs as 1D arrays (.npy files) that only write out the valid grid cell values for the target state.  This greatly improves storage size.  The arrays can then be converted to CSV file containing the following fields using the `convert_1d_array_to_csv()` function:  [Xcoord, Ycoord, FID, n], where "XCoord" is the X coordinate value, "YCoord" is the Y coordinate value, "FID" is the index of the grid cell when flattened to a 1D array, and "n" is the population output from the run.

### Future simulation

#### Running a simulation for each LHS sample using year 2010 as the base year (observed) and 2020 as the projected year using population projections from SSP2

**NOTE:** The following is meant to run on a cluster that utilizes the SLURM to schedule jobs.

```python
import sa_popgrid


output_script_dir = '<the path to the directory that my SLURM scripts will be written to>'

# the number of samples in your sample array
n_samples = 1000

# your directory to the newly modified inputs
data_dir = '<your directory that holds the newly modified inputs>'

# directory to write the outputs to
simulation_output_dir = '<your output dir>'

# directory to write the SLURM out log files to
slurm_out_dir = '<your output dir>'

# directory of my Python virtual environment
venv_dir = '/my/env'

# year to use as the historical reference data (e.g., observed data or previous year)
hist_yr = 2010

# validation or projection year
proj_yr = 2020

# state name to process as all lower case with underscore separators
state_name = '<my_target_state_name>'

# using validation as the scenario name
scenario = 'SSP2'

# see help for additional option explanations
sa_popgrid.submit_slurm_array(output_script_dir,
                               n_samples, data_dir,
                               simulation_output_dir,
                               slurm_out_dir,
                               venv_dir,
                               hist_yr,
                               proj_yr,
                               state_name,
                               scenario,
                               walltime='04:00:00',
                               submit_job=True,
                               partition='normal',
                               account_name='',
                               max_jobs=10,
                               method='simulation',
                               lhs_array_file=None,
                               lhs_problem_file=None,
                               write_raster=False,
                               output_total=False,
                               write_array1d=True)
```
