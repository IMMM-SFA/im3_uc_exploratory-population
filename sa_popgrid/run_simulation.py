import os
import pkg_resources

import simplejson
import numpy as np
import pandas as pd

import sa_popgrid.utils as utils
from population_gravity import Model


def run_simulation(grid_coordinates_file,
                    base_rural_pop_raster,
                    base_urban_pop_raster,
                    historical_suitability_raster,
                    projected_population_file,
                    one_dimension_indices_file,
                    output_directory,
                    alpha_urban,
                    alpha_rural,
                    beta_urban,
                    beta_rural,
                    kernel_distance_meters,
                    scenario,
                    state_name,
                    historic_base_year,
                    projection_year,
                    write_raster=True,
                    write_array1d=False,
                    write_array2d=False,
                    write_csv=False,
                    write_logfile=False,
                    write_suitability=False):

    """Run simuation to downscale population for a projected year."""

    run = Model(grid_coordinates_file=grid_coordinates_file,
                base_rural_pop_raster=base_rural_pop_raster,
                base_urban_pop_raster=base_urban_pop_raster,
                historical_suitability_raster=historical_suitability_raster,
                projected_population_file=projected_population_file,
                one_dimension_indices_file=one_dimension_indices_file,
                output_directory=output_directory,
                alpha_urban=alpha_urban,
                alpha_rural=alpha_rural,
                beta_urban=beta_urban,
                beta_rural=beta_rural,
                kernel_distance_meters=kernel_distance_meters,
                scenario=scenario,
                state_name=state_name,
                historic_base_year=historic_base_year,
                projection_year=projection_year,
                write_raster=write_raster,
                write_array1d=write_array1d,
                write_array2d=write_array2d,
                write_csv=write_csv,
                write_logfile=write_logfile,
                write_suitability=write_suitability)

    run.downscale()


def run_validation_allstates(historical_year, projection_year, data_dir, simulation_output_dir,
                             kernel_distance_meters=100000, scenario='validation'):
    """Run validation for all states. Validation run to compare observed versus simulated to see if we can reproduce
    the outputs that were published.  Population projections are taken directly from the year in which the
    validation is being conducted (e.g., if projection_year is 2010, 2010 projected population will be from the
    observed 2010 data sets). Calibration parameters are those that were used in the publication.

    :param historical_year:                     Base year of observed data in YYYY format
    :type historical_year:                      int

    :param projection_year:                     Validation year from observed data in YYYY format
    :type projection_year:                      int

    :param data_dir:                            Directory containing the input data from the newly formatted inputs.
    :type data_dir:                             str

    :param simulation_output_dir:               Output directory to save validation simulations to
    :type simulation_output_dir:                str

    :param kernel_distance_meters:              Kernel distance to search for nearby suitability
    :type kernel_distance_meters:               float

    :param scenario:                            Scenario name
    :type scenario:                             str

    """

    # get a list of all states to process; name all lower case and underscore separated where necessary
    state_list = utils.get_state_list()

    for target_state in state_list:

        run_validation(target_state,
                       historical_year,
                       projection_year,
                       data_dir,
                       simulation_output_dir,
                       kernel_distance_meters,
                       scenario)


def run_validation(target_state, historical_year, projection_year, data_dir, simulation_output_dir,
                   kernel_distance_meters=100000, scenario='validation'):
    """Validation run to compare observed versus simulated to see if we can reproduce the outputs that were
    published.  Population projections are taken directly from the year in which the validation is being
    conducted (e.g., if projection_year is 2010, 2010 projected population will be from the observed 2010 data sets).
    Calibration parameters are those that were used in the publication.

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

    :param kernel_distance_meters:              Kernel distance to search for nearby suitability
    :type kernel_distance_meters:               float

    :param scenario:                            Scenario name
    :type scenario:                             str

    """

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
                write_raster=True,
                write_logfile=True,
                output_total=True)

    run.downscale()
