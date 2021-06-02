import os
import pkg_resources

import numpy as np
import pandas as pd

import rasterio

from population_gravity import Model


def compare_outputs(original_raster, simulated_raster, relative_tolerance=1e-6, absolute_tolerance=1e-6, let_fail=True):
    """Compare the simulated outputs to what was published.  This function asserts equal array data structures and
    content.  Set relative and absolute tolerance to 1e-6 to cover version and OS inconsistency.

    :param original_raster:                 Full path with file name and extension to the original TIF raster
    :type original_raster:                  str

    :param simulated_raster:                Full path with file name and extension to the simulated TIF raster
    :type simulated_raster:                 str

    """

    with rasterio.open(original_raster) as orig:
        orig_array = orig.read(1)

    with rasterio.open(simulated_raster) as sim:
        sim_array = sim.read(1)

    if let_fail:
        np.testing.assert_allclose(orig_array, sim_array, rtol=relative_tolerance, atol=absolute_tolerance)

        return 'valid'

    else:
        try:
            np.testing.assert_allclose(orig_array, sim_array, rtol=relative_tolerance, atol=absolute_tolerance)
            return 'valid'

        except AssertionError:
            return 'invalid'


def get_state_list():
    """Get a list of states from the input directory."""

    states_df = pd.read_csv(pkg_resources.resource_filename('population_gravity', 'data/neighboring_states_100km.csv'))

    return states_df['target_state'].unique()


def compare_observed_versus_simulated(target_state, scenario, historical_year, projection_year, data_dir,
                                      simulation_output_dir, kernel_distance_meters=100000, let_fail=True):
    """Compare observed versus simulated to see if we can reproduce the outputs that were published."""

    # store test results in a list
    results = []

    input_dir = os.path.join(data_dir, target_state, 'inputs')
    original_output_dir = os.path.join(data_dir, target_state, 'outputs', 'model', scenario.upper())

    # the archived data left spaces in state names within files
    target_state_file = ' '.join(target_state.split('_'))

    grid_coordinates_file = os.path.join(input_dir, f'{target_state_file}_coordinates.csv')
    base_rural_pop_raster = os.path.join(input_dir, f'{target_state_file}_rural_{historical_year}_1km.tif')
    base_urban_pop_raster = os.path.join(input_dir, f'{target_state_file}_urban_{historical_year}_1km.tif')
    historical_suitability_raster = os.path.join(input_dir, f'{target_state_file}_mask_short_term.tif')
    projected_population_file = os.path.join(input_dir, f'{target_state_file}_{scenario}_popproj.csv')
    parameter_file = os.path.join(input_dir, f'{target_state_file}_{scenario}_params.csv')
    one_dimension_indices_file = os.path.join(input_dir, f'{target_state_file}_within_indices.txt')

    # create a data frame from the projected data
    params_df = pd.read_csv(parameter_file)

    # unpack parameters
    alpha_urban = params_df['Alpha_Urban'].values[0]
    alpha_rural = params_df['Alpha_Rural'].values[0]
    beta_urban = params_df['Beta_Urban'].values[0]
    beta_rural = params_df['Beta_Rural'].values[0]
    kernel_distance_meters = kernel_distance_meters

    # reproduce original data for California
    run = Model(grid_coordinates_file=grid_coordinates_file,
                base_rural_pop_raster=base_rural_pop_raster,
                base_urban_pop_raster=base_urban_pop_raster,
                historical_suitability_raster=historical_suitability_raster,
                projected_population_file=projected_population_file,
                one_dimension_indices_file=one_dimension_indices_file,
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

    # compare simulation outputs with originally published data
    original_rural_output_raster = os.path.join(original_output_dir, f'{target_state_file}_1km_{scenario}_rural_{projection_year}.tif')
    original_urban_output_raster = os.path.join(original_output_dir, f'{target_state_file}_1km_{scenario}_urban_{projection_year}.tif')
    original_total_output_raster = os.path.join(original_output_dir, f'{target_state_file}_1km_{scenario}_total_{projection_year}.tif')

    simulated_rural_output_raster = os.path.join(simulation_output_dir, f'{target_state}_1km_{scenario}_rural_{projection_year}.tif')
    simulated_urban_output_raster = os.path.join(simulation_output_dir, f'{target_state}_1km_{scenario}_urban_{projection_year}.tif')
    simulated_total_output_raster = os.path.join(simulation_output_dir, f'{target_state}_1km_{scenario}_total_{projection_year}.tif')

    # compare to 6 decimal points; units are in n humans
    result_rural = compare_outputs(simulated_rural_output_raster, original_rural_output_raster, let_fail=let_fail)
    results.append(f'{target_state},rural,{result_rural}\n')

    result_urban = compare_outputs(simulated_urban_output_raster, original_urban_output_raster, let_fail=let_fail)
    results.append(f'{target_state},urban,{result_urban}\n')

    result_total = compare_outputs(simulated_total_output_raster, original_total_output_raster, let_fail=let_fail)
    results.append(f'{target_state},total,{result_total}\n')

    return results


def reproduce_experiment(data_dir, simulation_output_dir, base_year, projection_year, scenario, let_fail=False):
    """Reproduce the experiment for a given year using the data archived from the following publication and run
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
    :type projection_year:                      int

    :param scenario:                            Scenario name, e.g. SSP2
    :type scenario:                             str

    :param let_fail:                            If True, the assertion test will raise an exception and report the
                                                discrepancy.  If False, the assertion will be logged as either 'valid'
                                                or 'invalid' and the code will continue
    :type let_fail:                             bool

    """

    # assertion log keeps track of what has passed and failed assertion for matching published results
    assertion_log = os.path.join(simulation_output_dir, 'assertion.log')

    # get a list of all states to process; name all lower case and underscore separated where necessary
    state_list = get_state_list()

    result_logs = []

    # run each state
    try:
        for i in state_list:

            # run the simulation for the target state and compare the outputs to what was published
            results = compare_observed_versus_simulated(i, scenario, base_year, projection_year, data_dir,
                                                        simulation_output_dir, let_fail=let_fail)

            result_logs.extend(results)

    finally:

        # write out assertion results
        with open(assertion_log, 'w') as log:

            # write header
            log.write('state_name,category,result\n')

            for i in result_logs:
                log.write(i)
