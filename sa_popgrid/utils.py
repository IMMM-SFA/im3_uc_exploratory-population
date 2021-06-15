import os
import pkg_resources

import rasterio
import simplejson
import numpy as np
import pandas as pd


def get_population_from_raster(raster_file, indices_list) -> float:
    """Get the population sum of all valid grid cells within a state.

    :param raster_file:             Full path with file name and extension to the input population raster file
    :type raster_file:              str

    :param indices_list:            List of index values for grid cells that are within the target state
    :type indices_list:             ndarray

    :return:                        population sum in number of humans for the target state

    """

    with rasterio.open(raster_file) as src:

        return src.read(1).flatten()[indices_list].sum()


def get_state_list():
    """Get a list of states from the input directory.

    :return:                                    Array of all states

    """

    states_df = pd.read_csv(pkg_resources.resource_filename('population_gravity', f'data/neighboring_states_150km.csv'))

    return states_df['target_state'].unique()


def convert_1d_array_to_csv(coordinate_file, indices_file, run_1d_array_file, output_dir, output_type='valid',
                            nodata=-3.4028235e+38):
    """Convert a 1D array of grid cells values that fall within the target state to
    a CSV file containing the following fields: [XCoord, YCoord, FID, n] where "XCoord" is
    the X coordinate value, "YCoord" is the Y coordinate value, "FID" is the index
    of the grid cell when flattened to a 1D array, and "n" is the population output from
    the run.

    The projected coordinate system is "EPSG:102003 - USA_Contiguous_Albers_Equal_Area_Conic"

    :param coordinate_file:                     Full path with file name and extension to the input coordinate CSV
                                                file containing all grid cell coordinates and their index ID for
                                                each grid cell in the sample space.  File name is generally:
                                                <state_name>_coordinates.csv
    :type coordinate_file:                      str

    :param indices_file:                        Full path with file name and extension to the input indicies file
                                                containing a list of index values that represent grid cells that fall
                                                inside the state boundary. These index values are used to extract grid
                                                cell values from rasters that have been read to a 2D array and then
                                                flattened to 1D; where they still contain out-of-bounds-data.  File name
                                                is generally:  <state_name>_within_indices.txt
    :type indices_file:                         str

    :param run_1d_array_file:                   Full path with file name and extension to a 1D array generated from
                                                a run output.  This array contains only grid cell values that fall
                                                within the boundary of the target state.  File name is generally:
                                                <state_name>_1km_<scenario>_<setting>_<year>_1d.npy; where `scenario` is
                                                the SSP and `setting` in either "rural", "urban", or "total"
    :type run_1d_array_file:                    str

    :param output_dir:                          Full path to the directory you wish to save the file in.
    :type output_dir:                           str


    :param output_type:                         Either "valid" or "full".  Use "valid" to only export the grid cells
                                                that are within the target state.  Use "full" to export all grid cells
                                                for the full extent.  Default:  "valid"
    :type output_type:                          str

    :param nodata:                              Value for NoData in the raster.  Default:  -3.4028234663852886e+38
    :type nodata:                               float

    :return:                                    Data Frame of the combined data that has been written to a CSV file

    """

    # validate output type
    output_type = output_type.lower()
    if output_type not in ('valid', 'full'):
        raise ValueError(f"`output_type` must be either 'valid' or 'full'.  You entered:  '{output_type}'")

    # read in coordinate file
    df_coords = pd.read_csv(coordinate_file)

    # read in within_indices file
    with open(indices_file, 'r') as rn:
        indices_list = simplejson.load(rn)

    # load run 1D array
    arr = np.load(run_1d_array_file)

    # create a data frame of the run data
    df_data = pd.DataFrame({'FID': indices_list, 'n': arr})

    # merge the data and the coordinates
    if output_type == 'valid':
        df_merge = pd.merge(df_coords, df_data, on='FID', how='inner')
    else:
        df_merge = pd.merge(df_coords, df_data, on='FID', how='left')

        # fill NaN for grid cells that are not in the target state with the nodata value
        df_merge.fillna(nodata, inplace=True)

    # save as CSV
    fname = f"{'_'.join(os.path.splitext(os.path.basename(f))[0].split('_')[:-1])}.csv"
    output_csv = os.path.join(output_dir, fname)
    df_merge.to_csv(output_csv, index=False)

    return df_merge


def convert_1d_to_2d_array(run_1d_array_file, indices_file, template_raster_file):
    """Convert a 1-dimension array of valid grid cells for a target state to a 2-dimension array containing to full extent.

    :param indices_file:                        Full path with file name and extension to the input indicies file
                                                containing a list of index values that represent grid cells that fall
                                                inside the state boundary. These index values are used to extract grid
                                                cell values from rasters that have been read to a 2D array and then
                                                flattened to 1D; where they still contain out-of-bounds-data.  File name
                                                is generally:  <state_name>_within_indices.txt
    :type indices_file:                         str

    :param run_1d_array_file:                   Full path with file name and extension to a 1D array generated from
                                                a run output.  This array contains only grid cell values that fall
                                                within the boundary of the target state.  File name is generally:
                                                <state_name>_1km_<scenario>_<setting>_<year>_1d.npy; where `scenario` is
                                                the SSP and `setting` in either "rural", "urban", or "total"
    :type run_1d_array_file:                    str

    :param template_raster_file:                Full path with file name and extension to a template raster file to use
                                                as a template. This file should be from the same source by which the
                                                1D run array was created.
    :type template_raster_file:                 str

    :return:                                    2D array

    """

    # load 1d array
    run_arr_1d = np.load(run_1d_array_file)

    # read in within_indices file
    with open(indices_file, 'r') as rn:
        indices_list = simplejson.load(rn)

    # read in template raster file
    with rasterio.open(template_raster_file) as src:
        template_arr_2d = src.read(1)
        template_arr_1d = template_arr_2d.flatten()
        metadata = src.meta.copy()

    # set template array values to nan
    template_arr_1d[:] = np.nan

    # replace template values with values from run
    template_arr_1d[indices_list] = run_arr_1d

    # reshape template to two dimensions
    return template_arr_1d.reshape(template_arr_2d.shape)


