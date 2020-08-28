import argparse
import os
import pickle
import numpy as np

from population_gravity import Model


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

    args = parser.parse_args()

    lhs_problem_file = os.path.join(args.sample_directory, f'lhs_{args.n_samples}_problem_dict.p')
    lhs_sample_file = os.path.join(args.sample_directory, f'lhs_{args.n_samples}_sample.npy')

    # load the sample files
    lhs_problem = pickle.load(open(lhs_problem_file, 'rb'))
    lhs_sample = np.load(lhs_sample_file)[args.sample_index]

    # get the parameter name list
    param_name_list = lhs_problem.get('names')

    # run the model for the target sample
    run = Model(grid_coordinates_file=os.path.join(args.input_directory, 'vermont_coordinates.csv'),
                historical_rural_pop_raster=os.path.join(args.input_directory, 'vermont_rural_2010_1km.tif'),
                historical_urban_pop_raster=os.path.join(args.input_directory, 'vermont_urban_2010_1km.tif'),
                historical_suitability_raster=os.path.join(args.input_directory, 'vermont_mask_short_term.tif'),
                projected_population_file=os.path.join(args.input_directory, 'vermont_SSP2_popproj.csv'),
                one_dimension_indices_file=os.path.join(args.input_directory, 'vermont_within_indices.txt'),
                output_directory=args.output_directory,
                alpha_urban=lhs_sample[param_name_list.index('alpha_urban')],
                alpha_rural=lhs_sample[param_name_list.index('alpha_rural')],
                beta_urban=lhs_sample[param_name_list.index('beta_urban')],
                beta_rural=lhs_sample[param_name_list.index('beta_rural')],
                kernel_distance_meters=lhs_sample[param_name_list.index('kernel_distance_meters')],
                scenario='SSP2',  # shared socioeconomic pathway abbreviation
                state_name='vermont',
                historic_base_year=2010,
                projection_start_year=2020,
                projection_end_year=args.end_yr,
                time_step=10,
                write_raster=False,
                write_array1d=True,
                write_array2d=False,
                write_suitability=True,
                write_csv=False,
                compress_csv=True,
                write_logfile=False,
                run_number=args.sample_index,
                output_total=False)

    run.downscale()
