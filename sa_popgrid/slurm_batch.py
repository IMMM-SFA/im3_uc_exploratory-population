import os


def submit_slurm_array(output_script_dir, data_dir, simulation_output_dir, slurm_out_dir, venv_dir, n_samples,
                       hist_yr, proj_yr, state_name, scenario, submit_job=True, walltime='01:00:00',
                       partition='normal', account_name='', jobname='batch', max_jobs=None, method='validation',
                       lhs_array_file=None,  lhs_problem_file=None, write_raster=False, output_total=False,
                       write_array1d=True):
    """This script will build a SLURM script, write it to the desired location, and submit the job using a SLURM array
    over the desired sample space.

    :param output_script_dir:                   Directory to write the output job scripts to
    :type output_job_script:                    str

    :param data_dir:                            Directory containing the input directory and associated state files
    :type data_dir:                             str

    :param simulation_output_dir:               Directory where the outputs will be stored
    :type simulation_output_dir:                str

    :param slurm_out_dir:                       Directory where the SLURM log files will be stored
    :type slurm_out_dir:                        str

    :param venv_dir:                            Full path to your Python virtual environment
    :type venv_dir:                             str

    :param n_samples:                           The number of samples in the sample array
    :param n_samples:                           int

    :param hist_yr:                             Base year of observed data in YYYY format
    :type hist_yr:                              int

    :param proj_yr:                             Projection or validation year in YYYY format
    :type proj_yr:                              int

    :param state_name:                          Target state name to run all lower case and underscore separated
    :type state_name:                           str

    :param scenario:                            Scenario name
    :type scenario:                             str

    :param submit_job:                          Choice to submit job to the SLURM queue after building the script
    :type submit_job:                           bool

    :param walltime:                            Time limit for each SLURM job in HH:MM:SS
                                                Corresponds to the '--time' SLURM setting
    :type walltime:                             str

    :param partition:                           Name of the partition to use on the cluster.
                                                Corresponds to the '--partition' SLURM setting
    :type partition:                            str

    :param account_name:                        If applicable, the name of the SLURM account to use.
                                                Corresponds to the '-A' SLURM setting
    :type account_name:                         str

    :param jobname:                             Name of the SLURM job to submit
                                                Corresponds to the '--job-name' SLURM setting
    :type jobname:                              str

    :param max_jobs:                            Maximum number of jobs to submit at any one time.
                                                Uses the '--array=0-100%10' setting where '%10' limits to 10 jobs in
                                                the SLURM queue at a time
    :type max_jobs:                             int

    :param method:                              Either 'validation' or 'simulation' which defines which function will
                                                be implemented in the SLURM script
    :type method:                               str

    :param lhs_array_file:                      Full path with file name and extension to the LHS array NPY file. If
                                                none is provide, the default 1000 samples file will be used.
    :type lhs_array_file:                       str

    :param lhs_problem_file:                    Full path with file name and extension to the LHS problem dictionary
                                                pickled file. If none is provide, the default 1000 samples file will be used.
    :type lhs_problem_file:                     str

    :param write_raster:                        Not optimal due to file size. Optionally export raster output
    :type write_raster:                         bool

    :param output_total:                        Not optimal due to file size. Choice to output total (urban + rural) dataset
    :type output_total:                         bool

    :param write_array1d:                       This choice is optimal to reduce file size
                                                outputs. Optionally export a Numpy 1D flattened array of only grid cells
                                                within the target state
    :type write_array1d:                        bool

    """

    method = method.lower()

    if method not in ('validation', 'simulation'):
        raise ValueError(f"Value for 'mode' = {method} is not valid. Must be 'validation' or 'simulation'")

    # set which function to use based off of the mode
    if method == 'validation':
        target_function = 'run_validation'
    else:
        target_function = 'run_simulation'

    # build strings for shell variables
    runtime_str = '{RUNTIME}'
    sample_index = '${SLURM_ARRAY_TASK_ID}'

    # apply the maximum number of jobs that can be in the queue at one time
    if max_jobs is None:
        job_limit = ''
    else:
        job_limit = f'%{max_jobs}'

    # set account name if provided
    if len(account_name) > 0:
        account = f'#SBATCH -A {account_name}'
    else:
        account = ''

    # negative indent to achieve correct formatting in job script file
    slurm = f"""#!/bin/sh
#SBATCH --partition={partition}
#SBATCH --nodes=1
#SBATCH --time={walltime}
#SBATCH --job-name={jobname}
#SBATCH --output={slurm_out_dir}/batch_{scenario}_{state_name}_{hist_yr}to{proj_yr}_job%a.out
{account}

# README -----------------------------------------------------------------------
#
# This script is auto-generated from the `sa_popgrid` Python package.
# This script will launch SLURM tasks that will execute batch.py to create
# run outputs for each sample and problem dictionary.
#
# To execute this script to create a sample set of 100 with only 15 jobs 
# allowed at a time execute the following:
#
# `sbatch --array=0-100%15 run_batch.sh`
#
# ------------------------------------------------------------------------------

# ensure we are pointing to GDAL libs correctly
source ~/.bash_profile

# activate Python virtual environment
source {venv_dir}/bin/activate

STARTTIME=`date +%s`

# execute Python script
python -c "import sa_popgrid; sa_popgrid.{target_function}(target_state='{state_name}',
                               historical_year={hist_yr},
                               projection_year={proj_yr},
                               data_dir='{data_dir}',
                               simulation_output_dir='{simulation_output_dir}',
                               scenario='{scenario}',
                               lhs_array_file={lhs_array_file},
                               lhs_problem_file={lhs_problem_file},
                               sample_id={sample_index},
                               write_logfile=False,
                               write_raster={write_raster},
                               output_total={output_total},
                               write_array1d={write_array1d})"

ENDTIME=`date +%s`
RUNTIME=$((ENDTIME-STARTTIME))

echo "Run completed in ${runtime_str} seconds." 

"""

    # construct file name for job script
    output_job_script = os.path.join(output_script_dir, f"batch_{scenario}_{state_name}_{hist_yr}to{proj_yr}.sh")

    with open(output_job_script, 'w') as out:
        out.write(slurm)

    if submit_job:
        cmd = f"sbatch --array=0-{n_samples-1}{job_limit} {output_job_script}"
        print(cmd)

        os.system(cmd)
