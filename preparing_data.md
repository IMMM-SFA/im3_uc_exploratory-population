## Conducting an original run

### Original data
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


### Prepare the data
Download and unzip the inputs and outputs as archived in Zoraghein and O'Neill (2020) from the following Zenodo archive:  [zoraghein-oneill_population_gravity_inputs_outputs.zip](https://zenodo.org/record/3756179/files/zoraghein-oneill_population_gravity_inputs_outputs.zip?download=1)

### Install the `population_gravity` package in a Python 3.6 or up environment:

This is the downscaling model.

```bash
python -m pip install git+https://github.com/IMMM-SFA/https://github.com/IMMM-SFA/population_gravity.git
```

### Install the `sa_popgrid` package in a Python 3.6 or up environment:

This is the code used to conduct analysis and run the model for evaluative purposes.

```bash
python -m pip install git+https://github.com/IMMM-SFA/sa_popgrid.git
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

### Reproduce the original experiment for year 2020
Zoraghein and O'Neill (2020) used 2010 as the base historical year.  Though the population projection files include 2010 as a projected year for each SSP.  The projected 2010 should not be confused with the actual observed population data in 2010; they are different.

The following runs assumes the base year of 2010 and uses the projected data for SSP2 for year 2020, along with it's published parameters, to produce a simulated run with the goal of seeing if we can reproduce the results from the experiement.

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

sa_popgrid.reproduce_experiment(data_dir, simulation_output_dir, base_year, projection_year, scenario, let_fail=False)
```

### Extending the input data to accommodate a larger kernel distance reach
The original data was created to only include grid cells from other states when running a target state for 100 km around the border of the target state.  Since we wanted to envelope the kernel distance parameter default setting of 100 km that was used in Zoraghein and O'Neill (2020) by 50 km (50 km to 150 km), we have to rebuild the original data to support this.

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

### Validation: Running a simulation using year 2000 as the observed data to simulate year 2010 observed data using published parameter values.  

Year 2000 was used as the base year to calibrate to year 2010 observed data.  We want to recreate the validation.


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

### Create Latin Hypercube Sample (LHS) and problem dictionary
The current LHS sample and problem dictionary that has been used for testing is stored within this package and can be accessed using:

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
