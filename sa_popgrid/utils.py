import pkg_resources

import pandas as pd


def get_state_list():
    """Get a list of states from the input directory.

    :return:                                    Array of all states

    """

    states_df = pd.read_csv(pkg_resources.resource_filename('population_gravity', f'data/neighboring_states_150km.csv'))

    return states_df['target_state'].unique()


def build_sbatch_call(output_job_script, samples):
    """Build the sbatch command that will be executed to submit a job.

    :param output_job_script:                   Full path with file name and extension to the output job script
    :type output_job_script:                    str

    :param samples:                             An integer or list of samples desired.  E.g., 1000 or [20, 50]
    :type samples:                              integer, list

    :return:                                    [0] sbatch submission string,
                                                [1] type of samples variable

    """

    type_sample_list = type(samples)

    if type_sample_list == int:
        sbatch_call = f"sbatch {output_job_script} {samples}"

    elif type_sample_list == list:
        len_samples = len(samples)

        if len_samples == 0:
            raise IndexError(f"`sample_list` must at least have one value.  E.g., [20]")

        elif len_samples == 1:
            sbatch_call = f"sbatch  {output_job_script} {samples[0]}"

        else:
            sample_string = ",".join(samples)
            sbatch_call = f"sbatch  --array={sample_string} {output_job_script}"

    else:
        sbatch_call = f"sbatch  --array={samples} {output_job_script}"

    return sbatch_call, type_sample_list
