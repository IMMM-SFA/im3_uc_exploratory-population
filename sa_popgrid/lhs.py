import argparse
import os

from population_gravity.sensitivity import Lhs


def main(n_samples, out_directory):

    # generate latin hypercube sample
    lhs = Lhs(alpha_urban_bounds=[-2.0, 2.0],
              alpha_rural_bounds=[-2.0, 2.0],
              beta_urban_bounds=[-2.0, 2.0],
              beta_rural_bounds=[-2.0, 2.0],
              kernel_distance_meters_bounds=[50000, 100000],
              n_samples=n_samples,
              problem_dict_outfile=os.path.join(out_directory, f'lhs_{n_samples}_problem_dict.p'),
              sample_outfile=os.path.join(out_directory, f'lhs_{n_samples}_sample.npy'))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('n_samples', metavar='n', type=int, help='an integer for the number of samples')
    parser.add_argument('output_directory', metavar='outdir', type=str, help='directory where the outputs will be written')

    args = parser.parse_args()

    main(args.n_samples, args.output_directory)
