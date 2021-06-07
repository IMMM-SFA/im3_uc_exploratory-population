import os


def run_batch(output_job_script, n_samples, data_dir, sample_directory, simulation_output_dir, slurm_out_dir,
              venv_dir, hist_yr, proj_yr, state_name, scenario, walltime='01:00:00', submit_job=True, partition='normal',
              account_name='', max_jobs=None, method='validation'):
    """This script will launch SLURM tasks that will execute batch.py to create run outputs for each
    sample and problem dictionary.

    :param output_job_script:                   Full path with file name and extension to the output job script
    :type output_job_script:                    str

    :param sample_list:                         A list of samples desired.  E.g., [20, 50]
    :type sample_list:                          list

    :param simulation_output_dir:               Full path to the directory where the outputs will be stored
    :type simulation_output_dir:                str

    :param venv_dir:                            Full path to your Python virtual environment
    :type venv_dir:                             str

    :param walltime:                            Time limit for each SLURM job in HH:MM:SS
    :type walltime:                             str

    :param method:                              Either 'validation' or 'simulation'
    :type method:                               str

    """

    method = method.lower()

    if method not in ('validation', 'simulation'):
        raise ValueError(f"Value for 'mode' = {method} is not valid. Must be 'validation' or 'simulation'")

    # set which function to use based off of the mode
    if method == 'validation':
        target_function = 'run_validation'

    # build strings for shell variables
    runtime_str = '{RUNTIME}'
    sample_index = '${SLURM_ARRAY_TASK_ID}'

    if max_jobs is None:
        job_limit = ''
    else:
        job_limit = f'%{max_jobs}'

    if len(account_name) > 0:
        account = f'#SBATCH -A {account_name}'
    else:
        account = ''

    slurm = f"""#!/bin/sh
                #SBATCH --partition={partition}
                #SBATCH --nodes=1
                #SBATCH --time={walltime}
                #SBATCH --job-name=batch
                #SBATCH --output=slurm_batch_ssp-{ssp}_state-{state_name}_yr-{end_yr}_sample-{sample_index}_job-%a.out
                {account}

                # README -----------------------------------------------------------------------
                #
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
                                               lhs_array_file=None,
                                               lhs_problem_file=None,
                                               sample_id={sample_index},
                                               write_logfile=False,
                                               write_raster=False,
                                               output_total=False,
                                               write_array1d=True)"

                ENDTIME=`date +%s`
                RUNTIME=$((ENDTIME-STARTTIME))

                echo "Run completed in ${runtime_str} seconds." """

    with open(output_job_script, 'w') as out:
        out.write(slurm)

    if submit_job:
        cmd = f"sbatch --array=0-{n_samples-1}{job_limit} {output_job_script}"
        print(cmd)

        os.system(cmd)
