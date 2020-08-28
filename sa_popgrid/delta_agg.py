import os
import argparse


def aggregate_outputs(output_dir, n_samples, setting, yr, state_name):

    # construct outfile name
    out_file = os.path.join(output_dir, f'delta-moment-independent_{state_name}_yr{yr}_{setting}_{n_samples}_per-gridcell.csv')

    # get a list of target files
    files = []
    for i in os.listdir(output_dir):

        if (f'_{state_name}_' in i) and (f'_{n_samples}samples_' in i) and (f'_{setting}_' in i) and (f'_yr{yr}_' in i):
            files.append(os.path.join(output_dir, i))


    with open(out_file, 'w') as out:
        for index, i in enumerate(files):

            with open(i) as get:
                for idx, line in enumerate(get):
                    # write header from first file
                    if (index == 0) and (idx == 0):
                        out.write(line)

                    elif (index > 0) and (idx == 0):
                        pass

                    else:
                        out.write(line)

            # delete individual file
            os.remove(i)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('n_samples', metavar='n', type=int, help='an integer for the number of samples')
    parser.add_argument('state_name', metavar='sn', type=str, help='name of the target state all lower case')
    parser.add_argument('output_directory', metavar='outdir', type=str, help='directory where the outputs will be written')
    parser.add_argument('setting', metavar='set', type=str, help='either Urban or Rural')
    parser.add_argument('target_year', metavar='yr', type=int, help='four digit target year')

    args = parser.parse_args()

    # aggregate individual outputs
    aggregate_outputs(args.output_directory, args.n_samples, args.setting, args.target_year, args.state_name)

