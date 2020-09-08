import os

import sa_popgrid.utils as utils


def run_lhs(output_job_script, samples, output_dir, venv_dir, alpha_urban_upper=2.0, alpha_urban_lower=-2.0,
            alpha_rural_upper=2.0, alpha_rural_lower=-2.0, beta_urban_upper=2.0, beta_urban_lower=-2.0,
            beta_rural_upper=2.0, beta_rural_lower=-2.0, kernel_density_lower=50000, kernel_density_upper=100000,
            walltime='00:10:00'):
    """Submit a SLURM job, or array of jobs, to generate a pickled problem dictionary and a NumPy
    array of samples.

    :param output_job_script:                   Full path with file name and extension to the output job script
    :type output_job_script:                    str

    :param samples:                             An integer or list of samples desired.  E.g., 1000 or [20, 50]
    :type samples:                              integer, list

    :param output_dir:                          Full path to the directory where the outputs will be stored
    :type output_dir:                           str

    :param venv_dir:                            Full path to your Python virtual environment
    :type venv_dir:                             str

    :param walltime:                            Time limit for each SLURM job in HH:MM:SS
    :type walltime:                             str


    Example:

    >>> import sa_popgrid
    >>>
    >>> # where to write the job script
    >>> output_job_script = '/home/fs02/pmr82_0001/spec994/projects/population/code'
    >>>
    >>> # run only one sample
    >>> sample_list = 1000
    >>>
    >>> # directory to store the output files in
    >>> output_dir = '/home/fs02/pmr82_0001/spec994/projects/population/outputs/lhs'
    >>>
    >>> # my Python virtual environemnt
    >>> venv_dir = '/home/fs02/pmr82_0001/spec994/pyenv'
    >>>
    >>> # submit the SLURM job
    >>> sa_popgrid.run_lhs(output_job_script,
    >>>                     sample_list,
    >>>                     output_dir,
    >>>                     venv_dir,
    >>>                     alpha_urban_upper=2.0,
    >>>                     alpha_urban_lower=-2.0,
    >>>                     alpha_rural_upper=2.0,
    >>>                     alpha_rural_lower=-2.0,
    >>>                     beta_urban_upper=2.0,
    >>>                     beta_urban_lower=-2.0,
    >>>                     beta_rural_upper=2.0,
    >>>                     beta_rural_lower=-2.0,
    >>>                     kernel_density_lower=50000,
    >>>                     kernel_density_upper=100000,
    >>>                     walltime='00:10:00')

    """

    # build strings for shell variables
    runtime_str = '${RUNTIME}'
    array_id_str = '${SLURM_ARRAY_TASK_ID}'
    first_param = '${1}'

    # build sbatch submission string
    sbatch_call = utils.build_sbatch_call(output_job_script, samples)

    # build slurm script
    slurm_array = f"""#!/bin/sh
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
                # ------------------------------------------------------------------------------
                
                # ensure we are pointing to GDAL libs correctly
                source ~/.bash_profile
                
                # activate Python virtual environment
                source {venv_dir}/bin/activate
                
                STARTTIME=`date +%s`
                
                # execute Python script
                python3 -c "from sa_popgrid.lhs import main; main({array_id_str}, '{output_dir}', {alpha_urban_upper}, {alpha_urban_lower}, {alpha_rural_upper}, {alpha_rural_lower}, {beta_urban_upper}, {beta_urban_lower}, {beta_rural_upper}, {beta_rural_lower}, {kernel_density_lower}, {kernel_density_upper})"
                
                ENDTIME=`date +%s`
                RUNTIME=$((ENDTIME-STARTTIME))
                
                echo "Run completed in {runtime_str} seconds." """

    slurm_job = f"""#!/bin/sh
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
                # To execute this script to create a sample set for 1000 samples execute the following:
                #
                # `sbatch run_lhs.sh 1000`
                #
                # ------------------------------------------------------------------------------

                # ensure we are pointing to GDAL libs correctly
                source ~/.bash_profile

                # activate Python virtual environment
                source {venv_dir}/bin/activate

                STARTTIME=`date +%s`

                # execute Python script
                python3 -c "from sa_popgrid.lhs import main; main({first_param}, '{output_dir}', {alpha_urban_upper}, {alpha_urban_lower}, {alpha_rural_upper}, {alpha_rural_lower}, {beta_urban_upper}, {beta_urban_lower}, {beta_rural_upper}, {beta_rural_lower}, {kernel_density_lower}, {kernel_density_upper})"

                ENDTIME=`date +%s`
                RUNTIME=$((ENDTIME-STARTTIME))

                echo "Run completed in {runtime_str} seconds." """

    # choose job type
    if type_sample_list == int:
        slurm = slurm_job
    else:
        slurm = slurm_array

    # write job file
    with open(output_job_script, 'w') as out:
        out.write(slurm)

    # submit job
    os.system(sbatch_call)
