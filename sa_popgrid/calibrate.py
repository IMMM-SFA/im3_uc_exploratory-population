"""
Created on Mon Dec 11 11:24:13 2017
A package that contains necessary functions for population downscaling.

@author: Hamidreza Zoraghein and Chris R. Vernon
"""

import argparse
import os
import time
import string
import pickle
import multiprocessing

import pandas as pd
import numpy as np
import scipy.optimize
import simplejson
import rasterio

from collections import deque
from scipy.spatial import cKDTree
from pathos.multiprocessing import ProcessingPool as Pool


def raster_to_array(raster):
    # Read the input raster before converting it to an array
    with rasterio.open(raster) as src_raster:
        band = src_raster.read(1)
        return band.flatten()


def array_to_raster(input_raster, input_array, within_indices, output_raster):
    # Read the template raster to be filled by output array values later
    with rasterio.open(input_raster) as src_raster:
        band = src_raster.read(1)
        src_profile = src_raster.profile
        row_count = band.shape[0]
        col_count = band.shape[1]
        flat_array = band.flatten()

    # Replace initial array values with those from the input array
    flat_array[within_indices] = input_array

    array = flat_array.reshape(row_count, col_count)

    with rasterio.open(output_raster, "w", **src_profile) as dst:
        dst.write_band(1, array)


def all_index_retriever(raster, columns):
    # Read the input raster before converting it to an array
    with rasterio.open(raster) as src_raster:
        array = src_raster.read(1)

    # Put the row, column and linear indices of all elements in a dataframe
    shape = array.shape
    index = pd.MultiIndex.from_product([range(s) for s in shape], names=columns)
    df = pd.DataFrame({'all_index': array.flatten()}, index=index).reset_index()
    df["all_index"] = df.index

    return df.astype({"row": np.int32, "column": np.int32, "all_index": np.int32})


def suitability_estimator(pop_dist_params):
    # id of the current focal point
    element = pop_dist_params[0]

    # Construct nearby indices
    neigh_indices = element + pop_dist_params[1]

    # Population of close points
    pop = pop_dist_params[2][neigh_indices]

    # Calculation of the other elements of the suitability equation; RuntimeWarning expected
    with np.errstate(divide='ignore'):
        pop_xx_alpha = np.power(pop, pop_dist_params[3])

        # correct div by 0
        pop_xx_alpha[pop_xx_alpha == np.inf] = 0

    pop_dist_factor = pop_xx_alpha * pop_dist_params[4]

    return np.sum(pop_dist_factor)


def dist_matrix_calculator(first_index, cut_off_meters, all_indices, coors_csv_file):
    # Read all points with their coordinates
    points = np.genfromtxt(coors_csv_file, delimiter=',', skip_header=1, usecols=(0, 1, 2), dtype=float)

    # Calculate distances between the first point and all other points within a 100km neighborhood
    cut_off_metres = cut_off_meters + 1
    tree_1 = cKDTree(points[first_index:first_index + 1, [0, 1]])
    tree_2 = cKDTree(points[:, [0, 1]])
    tree_dist = cKDTree.sparse_distance_matrix(tree_1, tree_2, cut_off_metres, output_type='dict', p=2)

    # Put distances and indices of neighboring in a dataframe
    dist_df = pd.DataFrame(columns=["near_id", "dis"])
    dist_df["near_id"] = points[list(zip(*tree_dist))[1], 2].astype(np.int32)
    dist_df["dis"] = tree_dist.values()
    dist_df = dist_df.loc[dist_df.loc[:, "dis"] != 0, :]  # Remove the distance to itself

    # Bring row and column indices of neighboring points by a join
    dist_df = dist_df.join(all_indices, on="near_id")

    # Add to columns holding the relative difference in rows and colums beween focal point and its neighbors
    foc_indices = all_indices.loc[first_index, ["row", "column"]].values
    dist_df["ind_diff"] = dist_df["near_id"] - first_index
    dist_df["row_diff"] = dist_df["row"] - foc_indices[0]
    dist_df["col_diff"] = dist_df["column"] - foc_indices[1]

    # Drop unwanted columns
    dist_df = dist_df.drop(["row", "column", "near_id", "all_index"], axis=1)

    dist_df = dist_df.astype({"ind_diff": np.int32, "row_diff": np.int32, "col_diff": np.int32})

    return dist_df


def pop_min_function(z, *params):
    # Initialize the parameters
    a, b = z

    # Inputs to the optimization
    setting = params[0]
    population_1st = params[1]  # Population of points in the first year (urban/rural)
    population_2nd = params[2]  # Population of points in the second year (urban/rural)
    total_population_1st = params[3]  # Total population of points in the first year
    points_mask = params[4]  # Mask values of points
    dist_matrix = params[5]  # Template distance matrix
    within_indices = params[6]  # Indices of points within the state boundary (subset of the above)

    # Outputs of the optimization at each step
    suitability_estimates = deque()  # Suitability estimates in the second year
    pop_estimates = np.zeros(len(within_indices))  # Population estimates in the second year

    # Calculate aggregate urban/rural population at times 1 and 2
    pop_t1 = population_1st[setting][within_indices].sum()
    pop_t2 = population_2nd[setting][within_indices].sum()

    # Population change between the two reference years
    pop_change = pop_t2 - pop_t1
    if pop_change < 0:
        negative_mod = 1
    else:
        negative_mod = 0

    # Differences in index between focal and nearby points as a template
    ind_diffs = dist_matrix["ind_diff"].values

    # Distances between current point and its close points
    ini_dist = dist_matrix["dis"].values / 1000.0
    dist = -b * ini_dist

    exp_xx_inv_beta_dist = np.exp(dist)

    # Initialize the parallelization
    pool = Pool(processes=multiprocessing.cpu_count())

    # Provide the inputs for the parallelized function
    parallel_elements = deque([(i, ind_diffs, total_population_1st, a, exp_xx_inv_beta_dist)
                               for i in within_indices])

    # Derive suitability estimates
    suitability_estimates = pool.map(suitability_estimator, parallel_elements)

    # Change suitability estimates to a numpy array
    suitability_estimates = np.array(suitability_estimates)

    # Extract only the necessary mask values that fall within the state boundary
    points_mask = points_mask[within_indices]

    # Population in the first year
    pop_first_year = population_1st[setting][within_indices]

    # In case of population decline, suitability estimates are reciprocated for non-zero values
    if negative_mod:

        # find those whose mask is 0 but have population, they should decline anyway
        mask_zero = np.where(points_mask == 0)[0]
        pop_non_zero = np.where(pop_first_year != 0)[0]

        # Those cells with mask value of zero and population are the intersection of the two above arrays
        pop_mask = np.intersect1d(mask_zero, pop_non_zero, assume_unique=True)

        # Change the mask value of the above cells to the mean so that they also lose population
        points_mask[pop_mask] = points_mask.mean()

        # Adjust suitability values by applying mask values
        suitability_estimates = points_mask * suitability_estimates

        # Inverse current mask values for a better reflection of population decline
        suitability_estimates[suitability_estimates != 0] = 1.0 / suitability_estimates[suitability_estimates != 0]

    else:
        # Adjust suitability values by applying mask values
        suitability_estimates = points_mask * suitability_estimates

    # Total suitability for the whole area, which is the summation of all individual suitability values
    tot_suitability = suitability_estimates.sum()

    # Final population estimate for each point if nagative mode is off
    pop_estimates = suitability_estimates / tot_suitability * pop_change + pop_first_year

    if negative_mod:
        while any(pop < 0 for pop in pop_estimates):  # To ensure there is no negative population
            new_tot_suitability = 0  # Total suitability calculated over points with positive population
            extra_pop_mod = 0  # For adjusting negative population values

            # Treating negative population values
            extra_pop_mod = abs(pop_estimates[pop_estimates < 0].sum())
            pop_estimates[pop_estimates < 0] = 0

            # Calculate the new total suitability value based on those points whose projected population is positive
            new_tot_suitability = suitability_estimates[pop_estimates > 0].sum()

            # Adjust non-negative population values to maintain the total aggregated population
            pop_estimates[pop_estimates > 0] = pop_estimates[pop_estimates > 0] - (
                        suitability_estimates[pop_estimates > 0] / new_tot_suitability) * extra_pop_mod

    # Produce the total error compared to observed values
    tot_error = abs(population_2nd[setting][within_indices] - pop_estimates).sum()

    return tot_error


def brute_optimization(params, a_lower=-2.0, a_upper=2.0, b_lower=-2.0, b_upper=2.0, n_alphas=8, n_betas=10):
    """This is the original optimization method that provided the initial optimization best guess.

    :param a_lower:                 initial alpha lower bound
    :param a_upper:                 initial alpha upper bound
    :param b_lower:                 initial beta lower bound
    :param b_upper:                 initial beta upper bound
    :param n_alphas:                number of alpha values to generate
    :param n_betas:                 number of beta values to generate


    """

    n_samples = n_alphas * n_betas

    bounds = ((a_lower, a_upper), (b_lower, b_upper))

    # Parameters to be used in optimization
    a_list = np.linspace(a_lower, a_upper, n_alphas)
    b_list = np.linspace(b_lower, b_upper, n_betas)

    # build initial data frame
    a_col = [a for a in a_list for b in b_list]
    b_col = [b for a in a_list for b in b_list]
    df = pd.DataFrame({'a': a_col, 'b': b_col})

    df['estimate'] = df.apply(lambda x: pop_min_function((x['a'], x['b']), *params), axis=1)

    return df, bounds, n_samples


def lhs_optimization(params, sample_file, problem_file, setting):
    """LHS sampling for initial optimization to provide a best guess."""

    # load sample from file
    lhs_arr = np.load(sample_file)

    n_samples = lhs_arr.shape[0]

    # load problem file to get bounds
    lhs_prob = pickle.load(open(problem_file, 'rb'))

    # get alpha (a) and beta (b) parameters depending upon setting
    if setting == 'Urban':
        a_col = [i[0] for i in lhs_arr]
        b_col = [i[2] for i in lhs_arr]
        bounds = (lhs_prob['bounds'][0], lhs_prob['bounds'][2])

    else:
        a_col = [i[1] for i in lhs_arr]
        b_col = [i[3] for i in lhs_arr]
        bounds = (lhs_prob['bounds'][1], lhs_prob['bounds'][3])

    df = pd.DataFrame({'a': a_col, 'b': b_col})

    df['estimate'] = df.apply(lambda x: pop_min_function((x['a'], x['b']), *params), axis=1)

    return df, bounds, n_samples


def calibration(urb_pop_fst_year, urb_pop_snd_year, rur_pop_fst_year, rur_pop_snd_year, mask_raster,
                region_code, ssp_code, point_indices, point_coors, datadir_output, kernel_distance_meters=100000,
                optimization_method='brute', sample_file=None, problem_file=None):

    print("Calibration begin")
    t0 = time.time()

    # Define local variables
    all_rasters     = {}  # Dictionary storing initial urban and rural population grids
    rur_pop_files   = []  # List containing rural population grids
    urb_pop_files   = []  # List containing urban population grids
    population_1st  = {}  # Dictionary containing population of each point in year 1
    population_2nd  = {}  # Dictionary containing population of each point in year 2
    parameters_dict = {}  # Dictionary storing urban/rural calibration parameters

    # Rural
    rur_pop_files.append(rur_pop_fst_year)
    rur_pop_files.append(rur_pop_snd_year)

    # Urban
    urb_pop_files.append(urb_pop_fst_year)
    urb_pop_files.append(urb_pop_snd_year)

    # Urban and rural
    all_rasters["Rural"] = rur_pop_files
    all_rasters["Urban"] = urb_pop_files

    # Populate the array containing mask values
    points_mask = raster_to_array(mask_raster)

    # Read historical urban and rural population grids into arrays
    for setting in all_rasters:
        # Create the dictionary containing population of each point in year 1
        population_1st[setting] = raster_to_array(all_rasters[setting][0])

        # Create the dictionary containing population of each point in year 2
        population_2nd[setting] = raster_to_array(all_rasters[setting][1])

    # Create an array containing total population values in the first historical year
    total_population_1st = population_1st["Rural"] + population_1st["Urban"]

    # All indices
    all_indices = all_index_retriever(urb_pop_snd_year, ["row", "column"])

    # Read indices of points that fall within the state boundary
    with open(point_indices, 'r') as r:
        within_indices = simplejson.load(r)

    # Calculate a distance matrix that serves as a template
    dist_matrix = dist_matrix_calculator(within_indices[0], kernel_distance_meters, all_indices, point_coors)

    # Parameter calculation for rural and urban
    for setting in all_rasters:

        print(f"Conducting initial optimization for {setting}...")
        t0_init = time.time()

        # parameters for minimization function
        params = (setting, population_1st, population_2nd, total_population_1st, points_mask, dist_matrix, within_indices)

        if optimization_method == 'brute':
            fst_results, bounds, n_samples = brute_optimization(params, a_lower=-2.0, a_upper=2.0, b_lower=-2.0, b_upper=2.0, n_alphas=8, n_betas=10)

        elif optimization_method == 'lhs':
            fst_results, bounds, n_samples = lhs_optimization(params, sample_file, problem_file, setting)

        print(f"Initial optimization completed for {setting} in {(time.time() - t0_init) / 60} minutes.")

        optimization_csv = os.path.join(datadir_output, f'optimization_results_{setting.lower()}_samples-{n_samples}_method-{optimization_method}.csv')
        print(f"Writing initial optimization for {setting} to '{optimization_csv}`")
        fst_results.to_csv(optimization_csv, index=False)

        # Use the point with the minimum value as an initial guess for the second optimizer
        (a0, b0) = fst_results.loc[fst_results["estimate"].idxmin(), ["a", "b"]]

        # Final optimization
        print(f"Executing final SciPy minimization for {setting}...")
        t0_final = time.time()
        parameters = scipy.optimize.minimize(pop_min_function, x0=(a0, b0), args=params, method="SLSQP", tol=0.001,
                                             options={"disp": True}, bounds=bounds)

        print(f"Final optimization completed for {setting} in {(time.time() - t0_final) / 60} minutes.")

        print(f"Final optimization outcome for {setting}:")
        print(parameters)
        parameters_dict[setting] = parameters["x"]

    # Write the parameters to the designated csv file
    out_param_file = os.path.join(datadir_output, f"{region_code}_{ssp_code}_samples-{n_samples}_method-{optimization_method}_parameters.csv")

    with open(out_param_file, 'w') as out:
        out.write("Region,SSP,Alpha_Rural,Beta_Rural,Alpha_Urban,Beta_Urban\n")
        out.write(f"{region_code},{ssp_code},{parameters_dict['Rural'][0]},{parameters_dict['Rural'][1]},{parameters_dict['Urban'][0]},{parameters_dict['Urban'][1]}\n")

    print(f"Final parameters written to '{out_param_file}`")

    print(f"Calibration completed in {(time.time() - t0) / 60}")


if __name__ == '__main__':

    # Run this code on the cube using:
    #  python calibrate.py 50 vermont SSP2 lhs /home/fs02/pmr82_0001/spec994/projects/population/inputs /home/fs02/pmr82_0001/spec994/projects/population/outputs/lhs /home/fs02/pmr82_0001/spec994/projects/population/outputs/calibration

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('n_samples', metavar='n', type=int, help='an integer for the number of samples')
    parser.add_argument('state_name', metavar='sn', type=str, help='name of the target state all lower case')
    parser.add_argument('ssp', metavar='ssp', type=str, help='name of the target ssp all upper case no spaces')
    parser.add_argument('optimization_method', metavar='method', type=str, help='name of the optimization method.  Either "lhs" or "brute"')
    parser.add_argument('input_directory', metavar='indir', type=str, help='directory where the population model inputs are kept')
    parser.add_argument('sample_directory', metavar='sampdir', type=str, help='directory where the sample and problem files are kept')
    parser.add_argument('output_directory', metavar='outdir', type=str, help='directory where the outputs will be written')

    args = parser.parse_args()

    sample_file = os.path.join(args.sample_directory, f"lhs_{args.n_samples}_sample.npy")
    problem_file = os.path.join(args.sample_directory, f"lhs_{args.n_samples}_problem_dict.p")

    urb_pop_fst_year = os.path.join(args.input_directory, f"{args.state_name}_urban_2000_1km.tif")
    urb_pop_snd_year = os.path.join(args.input_directory, f"{args.state_name}_urban_2010_1km.tif")
    rur_pop_fst_year = os.path.join(args.input_directory, f"{args.state_name}_rural_2000_1km.tif")
    rur_pop_snd_year = os.path.join(args.input_directory, f"{args.state_name}_rural_2010_1km.tif")
    mask_raster = os.path.join(args.input_directory, f"{args.state_name}_mask_short_term.tif")
    point_indices = os.path.join(args.input_directory, f"{args.state_name}_within_indices.txt")
    point_coors = os.path.join(args.input_directory, f"{args.state_name}_coordinates.csv")

    calibration(urb_pop_fst_year,
                urb_pop_snd_year,
                rur_pop_fst_year,
                rur_pop_snd_year,
                mask_raster,
                args.state_name,
                args.ssp,
                point_indices,
                point_coors,
                args.output_directory,
                optimization_method=args.optimization_method,
                sample_file=sample_file,
                problem_file=problem_file)
