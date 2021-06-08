import os
import pkg_resources
import pickle

import simplejson
import numpy as np
import pandas as pd

import sa_popgrid.utils as utils
from population_gravity import Model


def run_model(**kwargs):
    """Core simulation function to conduct either stand-alone or sample-based population_gravity model runs.

    If running for LHS samples, pass the alpha and beta parameter values and the default parameters will be overridden.

    :param target_state:                        Target state name to run all lower case and underscore separated
    :type target_state:                         str

    :param historical_year:                     Base year of observed data in YYYY format
    :type historical_year:                      int

    :param projection_year:                     Projection year in YYYY format
    :type projection_year:                      int

    :param data_dir:                            Directory containing the input data from the newly formatted inputs.
    :type data_dir:                             str

    :param simulation_output_dir:               Output directory to save simulations to
    :type simulation_output_dir:                str

    :param kernel_distance_meters:              Kernel distance to search for nearby suitability. This gets overridden
                                                when running LHS.
    :type kernel_distance_meters:               float

    :param scenario:                            Scenario name
    :type scenario:                             str

    :param lhs_array_file:                      Full path with file name and extension to the LHS array NPY file. If
                                                none is provide, the default 1000 samples file will be used.
    :type lhs_array_file:                       str

    :param lhs_problem_file:                    Full path with file name and extension to the LHS problem dictionary
                                                pickled file. If none is provide, the default 1000 samples file will be used.
    :type lhs_problem_file:                     str

    :param sample_id:                           Sample number from the LHS array by index.  If provided, the code will
                                                assume you are running in LHS mode.
    :type sample_id:                            int

    :param write_logfile:                       Optionally write log to file
    :type write_logfile:                        bool

    :param write_raster:                        Optionally export raster output
    :type write_raster:                         bool

    :param output_total:                        Choice to output total (urban + rural) dataset
    :type output_total:                         bool

    :param write_array1d:                       Optionally export a Numpy 1D flattened array of only grid cells
                                                within the target state
    :type write_array1d:                        bool

    """

    # unpack args
    target_state = kwargs.get('target_state')
    historical_year = kwargs.get('historical_year')
    projection_year = kwargs.get('projection_year')
    data_dir = kwargs.get('data_dir')
    simulation_output_dir = kwargs.get('simulation_output_dir')
    kernel_distance_meters = kwargs.get('kernel_distance_meters')
    scenario = kwargs.get('scenario')
    lhs_array_file = kwargs.get('lhs_array_file')
    lhs_problem_file = kwargs.get('lhs_problem_file')
    sample_id = kwargs.get('sample_id')
    write_logfile = kwargs.get('write_logfile')
    write_raster = kwargs.get('write_raster')
    output_total = kwargs.get('output_total')
    write_array1d = kwargs.get('write_array1d')

    input_dir = os.path.join(data_dir, target_state, 'inputs')

    # coordinates for each grid cell in the input rasters
    grid_coordinates_file = os.path.join(input_dir, f'{target_state}_coordinates.csv')

    # valid grid cell indices for a flattened array for the target state
    valid_indices_file = os.path.join(input_dir, f'{target_state}_within_indices.txt')

    # generate list of valid grid cell indices
    with open(valid_indices_file) as valid:
        valid_indices = np.array(simplejson.load(valid))

    # population rasters for the base year
    base_rural_pop_raster = os.path.join(input_dir, f'{target_state}_rural_{historical_year}_1km.tif')
    base_urban_pop_raster = os.path.join(input_dir, f'{target_state}_urban_{historical_year}_1km.tif')

    # suitability mask raster
    historical_suitability_raster = os.path.join(input_dir, f'{target_state}_mask_short_term.tif')

    # observed rasters for the validation year - this is not the projected raster from each scenario (e.g., SSP2, etc.)
    proj_rural_pop_raster = os.path.join(input_dir, f'{target_state}_rural_{projection_year}_1km.tif')
    proj_urban_pop_raster = os.path.join(input_dir, f'{target_state}_urban_{projection_year}_1km.tif')

    # get the projected population number from the observed data for the validation year
    rural_pop_proj_n = utils.get_population_from_raster(proj_rural_pop_raster, valid_indices)
    urban_pop_proj_n = utils.get_population_from_raster(proj_urban_pop_raster, valid_indices)

    # if not running LHS samples
    if sample_id == '':

        # retrieve the published calibration parameters for the target state and calibration year
        parameter_file = pkg_resources.resource_filename('sa_popgrid', f'data/calibration_parameters/{target_state}_calibration_params_{historical_year}to{projection_year}.csv')

        # create a data frame from the projected data
        params_df = pd.read_csv(parameter_file)

        # unpack parameters
        alpha_urban = params_df['Alpha_Urban'].values[0]
        alpha_rural = params_df['Alpha_Rural'].values[0]
        beta_urban = params_df['Beta_Urban'].values[0]
        beta_rural = params_df['Beta_Rural'].values[0]
        kernel_distance_meters = kernel_distance_meters

    else:

        # load lhs array from default if the user does not pass one
        if lhs_array_file is None:
            lhs_array = np.load(pkg_resources.resource_filename('sa_popgrid', 'data/lhs_1000_sample.npy'))
        else:
            lhs_array = np.load(lhs_array_file)

        # confirm the position of the parameters in the array
        if lhs_problem_file is None:

            with open(pkg_resources.resource_filename('sa_popgrid', 'data/lhs_1000_problem_dict.p'), "rb") as prob:
                lhs_problem_dict = pickle.load(prob)

        else:
            with open(lhs_problem_file, "rb") as prob:
                lhs_problem_dict = pickle.load(prob)

        # get parameter names in list
        param_names = lhs_problem_dict.get('names')

        # construct a data frame to query out target params
        df_params = pd.DataFrame({param_names[0]: lhs_array[:, 0],
                                  param_names[1]: lhs_array[:, 1],
                                  param_names[2]: lhs_array[:, 2],
                                  param_names[3]: lhs_array[:, 3],
                                  param_names[4]: lhs_array[:, 4]})

        # sample ids start with index of 0 and go through n_samples - 1
        df_params['n'] = df_params.index

        # get all parameters associated with a sample id
        x = df_params.loc[df_params['n'] == sample_id]

        alpha_urban = x['alpha_urban'].values[0]
        alpha_rural = x['alpha_rural'].values[0]
        beta_urban = x['beta_urban'].values[0]
        beta_rural = x['beta_rural'].values[0]
        kernel_distance_meters = x['kernel_distance_meters'].values[0]

    # reproduce original data
    run = Model(grid_coordinates_file=grid_coordinates_file,
                base_rural_pop_raster=base_rural_pop_raster,
                base_urban_pop_raster=base_urban_pop_raster,
                historical_suitability_raster=historical_suitability_raster,
                urban_pop_proj_n=urban_pop_proj_n,
                rural_pop_proj_n=rural_pop_proj_n,
                one_dimension_indices_file=valid_indices_file,
                output_directory=simulation_output_dir,
                alpha_urban=alpha_urban,
                alpha_rural=alpha_rural,
                beta_urban=beta_urban,
                beta_rural=beta_rural,
                kernel_distance_meters=kernel_distance_meters,
                scenario=scenario,
                state_name=target_state,
                historic_base_year=historical_year,
                projection_year=projection_year,
                write_raster=write_raster,
                write_logfile=write_logfile,
                output_total=output_total,
                write_array1d=write_array1d,
                run_number=sample_id)

    run.downscale()


def run_validation(target_state, historical_year, projection_year, data_dir, simulation_output_dir,
                   kernel_distance_meters=100000, scenario='validation', lhs_array_file=None,
                   lhs_problem_file=None, sample_id='', write_logfile=False, write_raster=False,
                   output_total=False, write_array1d=False):
    """Validation run to compare observed versus simulated to see if we can reproduce the outputs that were
    published.  Population projections are taken directly from the year in which the validation is being
    conducted (e.g., if projection_year is 2010, 2010 projected population will be from the observed 2010 data sets).
    Calibration parameters are those that were used in the publication.

    If running for LHS samples, pass the alpha and beta parameter values and the default parameters will be overridden.

    :param target_state:                        Target state name to run all lower case and underscore separated
    :type target_state:                         str

    :param historical_year:                     Base year of observed data in YYYY format
    :type historical_year:                      int

    :param projection_year:                     Validation year from observed data in YYYY format
    :type projection_year:                      int

    :param data_dir:                            Directory containing the input data from the newly formatted inputs.
    :type data_dir:                             str

    :param simulation_output_dir:               Output directory to save validation simulations to
    :type simulation_output_dir:                str

    :param kernel_distance_meters:              Kernel distance to search for nearby suitability. This gets overridden
                                                when running LHS.
    :type kernel_distance_meters:               float

    :param scenario:                            Scenario name
    :type scenario:                             str

    :param lhs_array_file:                      Full path with file name and extension to the LHS array NPY file. If
                                                none is provide, the default 1000 samples file will be used.
    :type lhs_array_file:                       str

    :param lhs_problem_file:                    Full path with file name and extension to the LHS problem dictionary
                                                pickled file. If none is provide, the default 1000 samples file will be used.
    :type lhs_problem_file:                     str

    :param sample_id:                           Sample number from the LHS array by index.  If provided, the code will
                                                assume you are running in LHS mode.
    :type sample_id:                            int

    :param write_logfile:                       Optionally write log to file
    :type write_logfile:                        bool

    :param write_raster:                        Optionally export raster output
    :type write_raster:                         bool

    :param output_total:                        Choice to output total (urban + rural) dataset
    :type output_total:                         bool

    :param write_array1d:                       Optionally export a Numpy 1D flattened array of only grid cells
                                                within the target state
    :type write_array1d:                        bool

    """

    run_model(target_state=target_state,
              historical_year=historical_year,
              projection_year=projection_year,
              data_dir=data_dir,
              simulation_output_dir=simulation_output_dir,
              kernel_distance_meters=kernel_distance_meters,
              scenario=scenario,
              lhs_array_file=lhs_array_file,
              lhs_problem_file=lhs_problem_file,
              sample_id=sample_id,
              write_logfile=write_logfile,
              write_raster=write_raster,
              output_total=output_total,
              write_array1d=write_array1d)


def run_validation_allstates(historical_year, projection_year, data_dir, simulation_output_dir,
                             kernel_distance_meters=100000, scenario='validation', lhs_array_file=None,
                             lhs_problem_file=None, sample_id='', write_logfile=False, write_raster=False,
                             output_total=False, write_array1d=False):
    """Run validation for all states for a single set of parameter values. Validation run to compare observed versus
    simulated to see if we can reproduce the outputs that were published.  Population projections are taken directly
    from the year in which the validation is being conducted (e.g., if projection_year is 2010, 2010 projected
    population will be from the observed 2010 data sets). Calibration parameters are those that were used in the
    publication.

    :param historical_year:                     Base year of observed data in YYYY format
    :type historical_year:                      int

    :param projection_year:                     Validation year from observed data in YYYY format
    :type projection_year:                      int

    :param data_dir:                            Directory containing the input data from the newly formatted inputs.
    :type data_dir:                             str

    :param simulation_output_dir:               Output directory to save validation simulations to
    :type simulation_output_dir:                str

    :param kernel_distance_meters:              Kernel distance to search for nearby suitability. This gets overridden
                                                when running LHS.
    :type kernel_distance_meters:               float

    :param scenario:                            Scenario name
    :type scenario:                             str

    :param lhs_array_file:                      Full path with file name and extension to the LHS array NPY file. If
                                                none is provide, the default 1000 samples file will be used.
    :type lhs_array_file:                       str

    :param lhs_problem_file:                    Full path with file name and extension to the LHS problem dictionary
                                                pickled file. If none is provide, the default 1000 samples file will be used.
    :type lhs_problem_file:                     str

    :param sample_id:                           Sample number from the LHS array by index.  If provided, the code will
                                                assume you are running in LHS mode.
    :type sample_id:                            int

    :param write_logfile:                       Optionally write log to file
    :type write_logfile:                        bool

    :param write_raster:                        Optionally export raster output
    :type write_raster:                         bool

    :param output_total:                        Choice to output total (urban + rural) dataset
    :type output_total:                         bool

    :param write_array1d:                       Optionally export a Numpy 1D flattened array of only grid cells
                                                within the target state
    :type write_array1d:                        bool

    """

    # get a list of all states to process; name all lower case and underscore separated where necessary
    state_list = utils.get_state_list()

    for target_state in state_list:

        run_validation(target_state=target_state,
                       historical_year=historical_year,
                       projection_year=projection_year,
                       data_dir=data_dir,
                       simulation_output_dir=simulation_output_dir,
                       kernel_distance_meters=kernel_distance_meters,
                       scenario=scenario,
                       lhs_array_file=lhs_array_file,
                       lhs_problem_file=lhs_problem_file,
                       sample_id=sample_id,
                       write_logfile=write_logfile,
                       write_raster=write_raster,
                       output_total=output_total,
                       write_array1d=write_array1d)


def run_simulation(target_state, historical_year, projection_year, data_dir, simulation_output_dir,
                   kernel_distance_meters=100000, scenario='SSP2', lhs_array_file=None,
                   lhs_problem_file=None, sample_id='', write_logfile=False, write_raster=False,
                   output_total=False, write_array1d=False):
    """Simulation run using the historical year raster as a starting point to downscale the population projection for
    the projection year.  Population projections are driven by scenario assumption for the SSP being evaluated.

    If running for LHS samples, pass the alpha and beta parameter values and the default parameters will be overridden.

    :param target_state:                        Target state name to run all lower case and underscore separated
    :type target_state:                         str

    :param historical_year:                     Base year of observed data in YYYY format
    :type historical_year:                      int

    :param projection_year:                     Projection year from scenario population input data in YYYY format
    :type projection_year:                      int

    :param data_dir:                            Directory containing the input data from the newly formatted inputs.
    :type data_dir:                             str

    :param simulation_output_dir:               Output directory to save simulations to
    :type simulation_output_dir:                str

    :param kernel_distance_meters:              Kernel distance to search for nearby suitability. This gets overridden
                                                when running LHS.
    :type kernel_distance_meters:               float

    :param scenario:                            Scenario name
    :type scenario:                             str

    :param lhs_array_file:                      Full path with file name and extension to the LHS array NPY file. If
                                                none is provide, the default 1000 samples file will be used.
    :type lhs_array_file:                       str

    :param lhs_problem_file:                    Full path with file name and extension to the LHS problem dictionary
                                                pickled file. If none is provide, the default 1000 samples file will be used.
    :type lhs_problem_file:                     str

    :param sample_id:                           Sample number from the LHS array by index.  If provided, the code will
                                                assume you are running in LHS mode.
    :type sample_id:                            int

    :param write_logfile:                       Optionally write log to file
    :type write_logfile:                        bool

    :param write_raster:                        Optionally export raster output
    :type write_raster:                         bool

    :param output_total:                        Choice to output total (urban + rural) dataset
    :type output_total:                         bool

    :param write_array1d:                       Optionally export a Numpy 1D flattened array of only grid cells
                                                within the target state
    :type write_array1d:                        bool

    """

    run_model(target_state=target_state,
              historical_year=historical_year,
              projection_year=projection_year,
              data_dir=data_dir,
              simulation_output_dir=simulation_output_dir,
              kernel_distance_meters=kernel_distance_meters,
              scenario=scenario,
              lhs_array_file=lhs_array_file,
              lhs_problem_file=lhs_problem_file,
              sample_id=sample_id,
              write_logfile=write_logfile,
              write_raster=write_raster,
              output_total=output_total,
              write_array1d=write_array1d)
