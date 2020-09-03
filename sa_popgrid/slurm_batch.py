import os


def run_batch(output_job_script, n_samples, input_directory, sample_directory, output_directory, end_yr,
              state_name, ssp, venv_dir, walltime='01:0:00', max_jobs=15):
    """This script will launch SLURM tasks that will execute batch.py to create run outputs for each
    sample and problem dictionary.

    :param output_job_script:                   Full path with file name and extension to the output job script
    :type output_job_script:                    str

    :param sample_list:                         A list of samples desired.  E.g., [20, 50]
    :type sample_list:                          list

    :param output_directory:                    Full path to the directory where the outputs will be stored
    :type output_directory:                     str

    :param venv_dir:                            Full path to your Python virtual environment
    :type venv_dir:                             str

    :param walltime:                            Time limit for each SLURM job in HH:MM:SS
    :type walltime:                             str

    """

    # build strings for shell variables
    runtime_str = '{RUNTIME}'
    sample_index = '${SLURM_ARRAY_TASK_ID}'

    slurm = f"""#!/bin/sh
                #SBATCH --partition=normal
                #SBATCH --nodes=1
                #SBATCH --ntasks=1
                #SBATCH --time={walltime}
                #SBATCH --job-name=batch
                #SBATCH --output=slurm_batch_%a.out

                # README -----------------------------------------------------------------------
                #
                # This script will launch SLURM tasks that will execute batch.py to create
                # run outputs for each sample and problem dictionary.
                #
                # To execute this script to create a sample set of 100 with only 10 jobs 
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
                python3 -c "from sa_popgrid.batch import main; main({n_samples}, '{input_directory}', '{sample_directory}', '{output_directory}', {sample_index}, {end_yr}, {state_name}, '{ssp}')"

                ENDTIME=`date +%s`
                RUNTIME=$((ENDTIME-STARTTIME))

                echo "Run completed in ${runtime_str} seconds." """

    with open(output_job_script, 'w') as out:
        out.write(slurm)

    os.system(f"sbatch --array=0-{n_samples-1}%{max_jobs} {output_job_script}")
