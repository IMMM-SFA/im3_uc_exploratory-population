import pkg_resources

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

def run_validation_allstates(**kwargs):
    """Run simulation to downscale population for a projected year for all states"""

    pass

def run_validation(target_state, historical_year, projection_year, data_dir,
                    simulation_output_dir, kernel_distance_meters=100000, let_fail=True):
    """Compare observed versus simulated to see if we can reproduce the outputs that were published."""

    input_dir = os.path.join(data_dir, target_state, 'inputs')

    # the archived data left spaces in state names within files
    target_state_file = ' '.join(target_state.split('_'))

    grid_coordinates_file = os.path.join(input_dir, f'{target_state}_coordinates.csv')
    base_rural_pop_raster = os.path.join(input_dir, f'{target_state}_rural_{historical_year}_1km.tif')
    base_urban_pop_raster = os.path.join(input_dir, f'{target_state}_urban_{historical_year}_1km.tif')
    historical_suitability_raster = os.path.join(input_dir, f'{target_state}_mask_short_term.tif')

    # TODO:  replace this with a function that gets the population from the projected years observed raster
    # TODO:  this solution is in get_observed_pop.ipynb
    projected_population_file = os.path.join(input_dir, f'{target_state}_{scenario}_popproj.csv')

    parameter_file = pkg_resources.resource_filename('sa_popgrid', 'data/calibration_parameters/f"{target_state}_calibration_params_{historical_year}to{projection_year}.csv")
    one_dimension_indices_file = os.path.join(input_dir, f'{target_state}_within_indices.txt')

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
