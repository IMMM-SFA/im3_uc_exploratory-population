import os
import argparse
import pickle
import time

import numpy as np
import pandas as pd
import multiprocessing
from SALib.analyze import delta
from pathos.multiprocessing import ProcessingPool as Pool

from dask_jobqueue import SLURMCluster
from distributed import Client, wait


class PrepSlurm:

    def __init__(self, valid_coordinates_csv, chunk_size=1000):

        self.valid_csv = valid_coordinates_csv
        self.chunk_size = chunk_size

        self.n_gridcells = self.get_n_gridcells()
        self.grid_list = list(range(self.n_gridcells))
        self.chunk_list = list(range(0, self.n_gridcells, self.chunk_size))

        self.chunk_generator = self.generate_chunk()

    def get_n_gridcells(self):
        """Get the number of grid cells in an input dataset."""

        # load the first file in the file list
        return pd.read_csv(self.valid_csv).shape[0]

    def generate_chunk(self):
        """Build a generator for chunks of grid cell indicies to process."""

        for index, i in enumerate(self.chunk_list):

            try:
                yield self.grid_list[i:self.chunk_list[index + 1]]

            except IndexError:
                yield self.grid_list[i:]

    def get_chunk(self):

        return next(self.chunk_generator)


class DeltaMomentIndependent:
    """Conduct analysis with the Delta Moment-Independent Measure using Latin Hypercube Sampling."""

    SETTING_OPTIONS = ('Urban', 'Rural')
    EXTENSION_OPTIONS = ('.tif', '.npy', '.csv', '.csv.gz')

    def __init__(self, problem_dict, sample, file_directory, setting, state_name, file_extension, output_directory,
                 target_year, output_type='_1d'):

        self.problem_dict = problem_dict
        self.sample = sample
        self._output_type = output_type
        self.target_year = str(target_year)

        self._file_directory = file_directory
        self._setting = setting
        self._state_name = state_name
        self._file_extension = file_extension
        self.output_directory = output_directory

        self.n_gridcells = self.get_n_gridcells()

    @property
    def suffix(self):
        """The string to search for in the file name that identifies the output type."""

        return self._output_type.lower()

    @property
    def file_extension(self):
        """Validate file extension."""

        if self._file_extension in self.EXTENSION_OPTIONS:
            return self._file_extension

        else:
            raise ValueError(
                f"Provided `file_extension` {self._file_extension} is not in the acceptable values:  {self.EXTENSION_OPTIONS}")

    @property
    def state_name(self):
        """Target state name."""

        return self._state_name.lower()

    @property
    def setting(self):
        """Either 'Urban' or 'Rural'"""

        if self._setting in self.SETTING_OPTIONS:
            return self._setting

        else:
            raise ValueError(
                f"Provided `setting` '{self._setting}' is not in the acceptable values:  {self.SETTING_OPTIONS}")

    @property
    def file_directory(self):
        """Validate directory of input files from the model run."""

        if os.path.isdir(self._file_directory):
            return self._file_directory

        else:
            raise NotADirectoryError(f"File directory '{self._file_directory} does not exist.")

    @property
    def file_list(self):
        """Get a list of files to process."""

        files = [os.path.join(self.file_directory, i) for i in os.listdir(self.file_directory) if
                 (i.split('_')[0] == self.state_name) and
                 (self.file_extension in i) and
                 (self.setting in i) and
                 (self.suffix in i) and
                 (self.target_year in i)]

        return self.validate_list(files)

    def get_n_gridcells(self):
        """Get the number of grid cells in an input dataset."""

        # load the first file in the file list
        return self.load_file(self.file_list[0]).shape[0]

    @property
    def data_array(self):
        """Combine output files into a single array.

        :return:                Array; shape = (n_runs, grid cells)

        """

        # shape (n_runs, grid cells)
        arr = np.zeros(shape=(len(self.file_list), self.n_gridcells))

        for index, i in enumerate(self.file_list):
            arr[index, :] = self.load_file(i)

        return arr

    def load_file(self, file):
        """Load data from file."""

        if self.file_extension == '.npy':
            return np.load(file)

        elif self.file_extension == '.csv.gz':
            return pd.read_csv(file, compression='gzip', sep=',')['value'].values

        else:
            raise ValueError(f"Loading '{self.file_extension}' is under development")

    def validate_list(self, in_list):
        """Ensure a list has a length > 0."""

        if len(in_list) > 0:
            return in_list

        else:
            raise ValueError(
                f"There are no files that match the search criteria of `state_name`='{self.state_name}', `file_extension`='{self.file_extension}', and `setting`='{self.setting}' in the file directory: '{self.file_directory}'")

    def run_slurm(self, chunk):
        """Run the sensitivity analysis and write the outputs to a file."""

        # get cpu count
        ncpus = multiprocessing.cpu_count()
        print(f"Number of CPUs:  {ncpus}")

        pool = Pool(processes=ncpus)
        results = pool.map(self.delta_gridcell, [i for i in chunk])

        #        results = []
        #        for i in chunk:
        #            results.append(self.delta_gridcell(i))

        # write results to file
        self.write_output(results, chunk)

    def delta_gridcell(self, i):
        """Generate statistics for a gridcell from a n-dim array."""
        from SALib.analyze import delta

        out_list = []

        # evaluate
        y = self.data_array[:, i]

        # if all values are the same
        unique_vals = np.unique(y).shape[0]

        if unique_vals > 1:

            try:
                # generate the sensitivity indices
                si = delta.analyze(self.problem_dict, self.sample, y, print_to_console=False)

                # write evaluated parameters
                for idx, key in enumerate(self.problem_dict['names']):
                    out_list.append(
                        f"{key},{si['delta'][idx]},{si['delta_conf'][idx]},{si['S1'][idx]},{si['S1_conf'][idx]},{i}\n")

            except(Exception) as e:

                # write evaluated parameters
                for idx, key in enumerate(self.problem_dict['names']):
                    out_list.append(f"{key},{np.nan},{np.nan},{np.nan},{np.nan},{i}\n")

        else:

            # write evaluated parameters
            for idx, key in enumerate(self.problem_dict['names']):
                out_list.append(f"{key},{np.nan},{np.nan},{np.nan},{np.nan},{i}\n")

        return out_list

    def write_output(self, result_list, chunk):
        """Write output to file."""

        output_file = os.path.join(self.output_directory,
                                   f"delta-moment-independent_{self.state_name}_yr{self.target_year}_{self.setting}_{self.sample.shape[0]}samples_gridcells-{min(chunk)}-{max(chunk)}.csv")

        with open(output_file, 'w') as out:

            # write header for output file
            out.write('param,delta,delta_conf,S1,S1_conf,gridcell\n')

            for element in result_list:
                for param in element:
                    out.write(param)


if __name__ == '__main__':
    # Run this code by executing the following in a terminal after activating your Python virtual environment:
    #   python delta_moment.py 50  vermont /home/fs02/pmr82_0001/spec994/projects/population/inputs /home/fs02/pmr82_0001/spec994/projects/population/outputs/lhs /home/fs02/pmr82_0001/spec994/projects/population/outputs/batch  /home/fs02/pmr82_0001/spec994/projects/population/outputs/delta  8  Urban _1d 2020

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('n_samples', metavar='n', type=int, help='an integer for the number of samples')
    parser.add_argument('state_name', metavar='sn', type=str, help='name of the target state all lower case')
    parser.add_argument('input_directory', metavar='indir', type=str,
                        help='directory where the population model inputs are kept')
    parser.add_argument('sample_directory', metavar='sampdir', type=str,
                        help='directory where the sample and problem files are kept')
    parser.add_argument('run_directory', metavar='rundir', type=str, help='directory where the batch runs are kept')
    parser.add_argument('output_directory', metavar='outdir', type=str,
                        help='directory where the outputs will be written')
    parser.add_argument('chunk_size', metavar='chk', type=int,
                        help='an integer for the number of grid cells to process at one time')
    parser.add_argument('setting', metavar='set', type=str, help='either Urban or Rural')
    parser.add_argument('output_type', metavar='otype', type=str, help='either suitability or _1d')
    parser.add_argument('target_year', metavar='yr', type=int, help='four digit target year')

    args = parser.parse_args()

    print(args)

    #    valid_csv = os.path.join(args.input_directory, f"{args.state_name}_valid_coordinates.csv")

    #    tasks = PrepSlurm(valid_csv, chunk_size=args.chunk_size)

    lhs_problem_file = os.path.join(args.sample_directory, f"lhs_{args.n_samples}_problem_dict.p")
    lhs_sample_file = os.path.join(args.sample_directory, f"lhs_{args.n_samples}_sample.npy")

    file_extension = '.npy'

    pdict = pickle.load(open(lhs_problem_file, 'rb'))
    samp = np.load(lhs_sample_file)

    delta_run = DeltaMomentIndependent(problem_dict=pdict,
                                       file_directory=args.run_directory,
                                       state_name=args.state_name,
                                       sample=samp,
                                       setting=args.setting,  # either 'Urban' or 'Rural'
                                       file_extension=file_extension,
                                       # file extension matching the output format from run files
                                       target_year=args.target_year,
                                       output_directory=args.output_directory,
                                       output_type=args.output_type)
    # set up cluster
    with SLURMCluster(
            queue='normal',
            cores=16,
            processes=1,
            walltime="01:00:00",
            memory='128 GB') as cluster:
        cluster.adapt(minimum_jobs=15, maximum_jobs=20)

        with Client(cluster) as client:
            # subs = [tasks.get_chunk() for i in tasks.chunk_list]
            n_gridcells = 24905
            l = list(range(n_gridcells))
            subs = [l[i:i + args.chunk_size] for i in range(0, n_gridcells, args.chunk_size)]

            t0 = time.time()

            futures = client.map(delta_run.run_slurm, subs)

            wait(futures)

            print(f"Runs completed in:  {(time.time() - t0) / 60} minutes.")
