import os


def run_lhs(job_script='', sample_string='20,50', output_dir='', venv_dir='', walltime='00:10:00'):

    array_id_str = '{SLURM_ARRAY_TASK_ID}'
    runtime_str = '{RUNTIME}'

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

    with open(job_script, 'w') as out:
        out.write(slurm)

    os.system(f"sbatch --array={sample_string} {job_script}")
