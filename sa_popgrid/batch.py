import argparse
import os
import pickle
import numpy as np

from population_gravity import Model


def main(n_samples, input_directory, sample_directory, output_directory, sample_index, end_yr, state_name, ssp):
    """Create run outputs for each sample and problem dictionary.

    :param n_samples:
    :param input_directory:
    :param sample_directory:
    :param output_directory:
    :param sample_index:
    :param end_yr:
    :param state_name:
    :param ssp:
    :return:
    """

    lhs_problem_file = os.path.join(sample_directory, f'lhs_{n_samples}_problem_dict.p')
    lhs_sample_file = os.path.join(sample_directory, f'lhs_{n_samples}_sample.npy')

    # load the sample files
    lhs_problem = pickle.load(open(lhs_problem_file, 'rb'))
    lhs_sample = np.load(lhs_sample_file)[sample_index]

    # get the parameter name list
    param_name_list = lhs_problem.get('names')

    # run the model for the target sample
    run = Model(grid_coordinates_file=os.path.join(input_directory, f'{state_name}_coordinates.csv'),
                historical_rural_pop_raster=os.path.join(input_directory, f'{state_name}_rural_2010_1km.tif'),
                historical_urban_pop_raster=os.path.join(input_directory, f'{state_name}_urban_2010_1km.tif'),
                historical_suitability_raster=os.path.join(input_directory, f'{state_name}_mask_short_term.tif'),
                projected_population_file=os.path.join(input_directory, f'{state_name}_{ssp}_popproj.csv'),
                one_dimension_indices_file=os.path.join(input_directory, f'{state_name}_within_indices.txt'),
                output_directory=output_directory,
                alpha_urban=lhs_sample[param_name_list.index('alpha_urban')],
                alpha_rural=lhs_sample[param_name_list.index('alpha_rural')],
                beta_urban=lhs_sample[param_name_list.index('beta_urban')],
                beta_rural=lhs_sample[param_name_list.index('beta_rural')],
                kernel_distance_meters=lhs_sample[param_name_list.index('kernel_distance_meters')],
                scenario=ssp,  # shared socioeconomic pathway abbreviation
                state_name=state_name,
                historic_base_year=2010,
                projection_start_year=2020,
                projection_end_year=end_yr,
                time_step=10,
                write_raster=False,
                write_array1d=True,
                write_array2d=False,
                write_suitability=True,
                write_csv=False,
                compress_csv=True,
                write_logfile=False,
                run_number=sample_index,
                output_total=False)

    run.downscale()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('n_samples', metavar='n', type=int, help='an integer for the number of samples')
    parser.add_argument('input_directory', metavar='indir', type=str,
                        help='directory where the population model inputs are kept')
    parser.add_argument('sample_directory', metavar='sampdir', type=str,
                        help='directory where the sample and problem files are kept')
    parser.add_argument('output_directory', metavar='outdir', type=str,
                        help='directory where the outputs will be written')
    parser.add_argument('sample_index', metavar='idx', type=int, help='index number of the sample')
    parser.add_argument('end_yr', metavar='eyr', type=int, help='four digit end year')
    parser.add_argument('state_name', metavar='sn', type=str, help='state name all lower case and underscore separated')
    parser.add_argument('ssp', metavar='ssp', type=str, help='Shared Socioeconomic Pathway abbreviation (e.g., SSP2)')

    args = parser.parse_args()

    main(args.n_samples,
         args.input_directory,
         args.sample_directory,
         args.output_directory,
         args.sample_index,
         args.end_yr,
         args.state_name,
         args.ssp)
