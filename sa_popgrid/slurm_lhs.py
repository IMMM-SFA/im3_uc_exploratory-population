import os


def run_lhs(output_job_script, sample_list, output_dir, venv_dir, walltime='00:10:00'):
    """Submit a SLURM job, or array of jobs, to generate a pickled problem dictionary and a NumPy
    array of samples.

    :param output_job_script:                   Full path with file name and extension to the output job script
    :type output_job_script:                    str

    :param sample_list:                         A list of samples desired.  E.g., [20, 50]
    :type sample_list:                          list

    :param output_dir:                          Full path to the directory where the outputs will be stored
    :type output_dir:                           str

    :param venv_dir:                            Full path to your Python virtual environment
    :type venv_dir:                             str

    :param walltime:                            Time limit for each SLURM job in HH:MM:SS
    :type walltime:                             str


    """

    # build strings for shell variables
    array_id_str = '{SLURM_ARRAY_TASK_ID}'
    runtime_str = '{RUNTIME}'

    # build sample string
    if type(sample_list) != list:
        raise TypeError(f"`sample_list` must be a list.  Your value:  {sample_list}")

    len_samples = len(sample_list)
    if len_samples == 0:
        raise IndexError(f"`sample_list` must at least have one value.  E.g., [20]")

    elif len_samples == 1:
        sample_string = f"{sample_list[0]}"

    else:
        sample_string = ",".join(sample_list)

    slurm = f"""#!/bin/sh
                #SBATCH --partition=normal
                #SBATCH --nodes=1
                #SBATCH --ntasks=1
                #SBATCH --time={walltime}
                #SBATCH --job-name=lhs
                #SBATCH --output=slurm_lhs_%a.out
                
                # README -----------------------------------------------------------------------
                #
                # This script will launch SLURM tasks that will execute lhs.py to create
                # samples and a problem dictionary for each array value passed.
                #
                # To execute this script to create a sample set for 20, and 50
                # samples execute the following:
                #
                # `sbatch --array=20,50 run_lhs.sh`
                #
                # To execute this script to build samples for 10 through 100 by 10:
                #
                # `sbatch --array=10-100:10 run_lhs.sh`
                #
                # This script currently assumes that there is all data is contained in a
                # directory within the users HOME directory named `projects/population`
                #
                # ------------------------------------------------------------------------------
                
                # ensure we are pointing to GDAL libs correctly
                source ~/.bash_profile
                
                # activate Python virtual environment
                source {venv_dir}/bin/activate
                
                STARTTIME=`date +%s`
                
                # execute Python script
                python3 -c 'from sa_popgrid.lhs import main; main(${array_id_str} ${output_dir})'
                
                ENDTIME=`date +%s`
                RUNTIME=$((ENDTIME-STARTTIME))
                
                echo "Run completed in ${runtime_str} seconds."""

    with open(output_job_script, 'w') as out:
        out.write(slurm)

    os.system(f"sbatch --array={sample_string} {output_job_script}")
