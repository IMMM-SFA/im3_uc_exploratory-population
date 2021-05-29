import time
import os
import shutil
import numpy as np
import pandas as pd
import geopandas as gpd
import simplejson
from zipfile import ZipFile
import zipfile
import rasterio
from rasterio.merge import merge
from rasterio.plot import show
from rasterio.plot import show_hist
from raster2xyz.raster2xyz import Raster2xyz
import georasters as gr
from unittest import TestCase as test


# directory of the original inputs and outputs
data_dir = "/Users/d3y010/projects/population/zoraghein-oneill_population_gravity_inputs_outputs"

# directory for the resolution corrected rasters to be written
template_directory = "/Users/d3y010/Desktop/templates"

# directory for the temporary outputs to be written
output_directory = '/Users/d3y010/Desktop/staging'

# file containing the neighboring states within 150 km from the target state
neighbor_state_file = '/Users/d3y010/repos/github/population_gravity/population_gravity/data/neighboring_states_150km.csv'

# temporary directory to store zipped files
temp_dir = '/Users/d3y010/Desktop/zips'


def get_neighbors(neighbors, state_name):
    """Get all neighboring states."""

    df = pd.read_csv(neighbors)
    states = df.loc[df['target_state'] == state_name]['near_state'].values

    return states


def set_affine_resolution(raster_file, out_directory, res=1000):
    """Fix rounding issues with the resolution of some rasters and save it to a temporary directory."""

    with rasterio.open(raster_file) as rx:
        # copy raster metadata
        rx_meta = rx.meta.copy()

        # get existing affine transformation
        a, b, c, d, e, f, g, h, i = list(rx_meta['transform'])

        # nudge resolution to our target 1000 meters
        adj_affine = rasterio.Affine(res, b, c, d, -res, f)

        # update the metadata object with the new transformation
        rx_meta.update({"transform": adj_affine})

        # write to file
        out_rx = os.path.join(out_directory, os.path.basename(raster_file))

        with rasterio.open(out_rx, 'w', **rx_meta) as dest:
            dest.write(rx.read())

        return out_rx


def create_template_raster(raster_list, out_file, set_value=0):
    """Create template raster mosaic set to value 0."""

    # get the metadata from the first raster in the list
    with rasterio.open(raster_list[0]) as rast:
        metadata = rast.meta.copy()

    # create mosaicked raster
    src_metadata = mosaic(raster_list, out_file, metadata, set_value)


def rasterio_assert(orig_file, new_file, orig_within_indices, new_within_indices):
    with rasterio.open(orig_file) as orig:
        orig_arr = orig.read(1).flatten()
        orig_within = orig_arr[orig_within_indices]

    with rasterio.open(new_file) as new:
        new_arr = new.read(1).flatten()
        new_within = new_arr[new_within_indices]

    np.testing.assert_array_equal(orig_within, new_within)


def mosaic(raster_list, out_raster, source_metadata, set_value=None, method='first'):
    """Create a raster mosiac from multiple rasters and save to file.

    :param raster_list:             List of full path to rasters with file name and extensions
    :param out_raster:              Full path with file name and extension to write the raster to
    :param source_metadata:         Metadata rasterio object from the target states init raster

    :return:                        Mosaicked rasterio object

    """
    # build list of raster objects
    raster_objects = [rasterio.open(i) for i in raster_list]

    # create mosaic
    mrg, out_transform = merge(raster_objects, method=method)

    # update metadata with mosiac values
    source_metadata.update({"height": mrg.shape[1], "width": mrg.shape[2], "transform": out_transform})

    # write output
    with rasterio.open(out_raster, 'w', **source_metadata) as dest:

        if set_value is None:
            dest.write(mrg)

        else:
            arr = mrg * 0 + set_value
            dest.write(arr)

    return source_metadata


def georaster_df_to_tiff(source_metadata, df, output_file):
    """Convert a Pandas data frame resulting from a Georasters conversion that contains
    row, col, value, x, and y fields to a GeoTiff."""

    # create a 2D array in the shape of the raster
    arr = np.reshape(df['value'].astype(np.float32).values, (df.row.max() + 1, df.col.max() + 1))

    # save as raster
    with rasterio.open(output_file, 'w', **source_metadata) as dest:
        dest.write(arr, 1)

    return arr


def create_template_dataframe(template_raster):
    # read in mosaic footprint having all 0 values
    template_df = gr.from_file(template_raster).to_pandas()

    # create primary key
    template_df['key'] = template_df['x'].round(4).astype(str) + '_' + template_df['y'].round(4).astype(str)

    return template_df


def mosaic_tile(template_df, raster_file):
    """Mosiac individual raster file onto a template data frame."""

    # overwrite template values with those in the individual raster
    df = gr.from_file(raster_file).to_pandas()

    # create key
    df['key'] = df['x'].round(4).astype(str) + '_' + df['y'].round(4).astype(str)

    # set key as index
    df.set_index('key', inplace=True)

    # convert to dictionary containing {key: value, ...}
    d = df[['value']].to_dict()['value']

    # overwrite the template raster value if the new value is greater
    template_df['new_value'] = template_df['key'].map(d)

    template_df['value'] = np.where(template_df['new_value'] > template_df['value'],
                                    template_df['new_value'],
                                    template_df['value'])

    # drop new value column
    template_df.drop(columns=['new_value'], inplace=True)

    return template_df


def mosaic_all(template_metadata, template_df, raster_list, out_file):
    """Mosaic a list of rasters onto a template raster and write the output to a file.

    :return:                  2D NumPy array

    """

    # add the incoming state population to the template
    for i in raster_list:
        template_df = mosaic_tile(template_df, i)

    # write the mosaic as a GeoTIFF
    return georaster_df_to_tiff(template_metadata, template_df, out_file)


def batch(data_dir, state_name, template_directory, output_directory, neighbor_state_file, temp_dir):

    # original coordinates file
    orig_coord_file = os.path.join(data_dir, state_name, "inputs", f"{state_name}_coordinates.csv")

    # new coordinates file
    new_coord_file = os.path.join(output_directory, f'{state_name}_coordinates.csv')

    # original within_indices file
    orig_indices_file = os.path.join(data_dir, state_name, "inputs", f"{state_name}_within_indices.txt")

    # new within_indices file
    new_within_indices_file = os.path.join(output_directory, f"{state_name}_within_indices.txt")

    # original suitability mask file
    corrected_mask_file = os.path.join(template_directory, f"{state_name}_mask_short_term.tif")

    # new suitability mask file that has been mosaicked
    out_mask_raster_file = os.path.join(output_directory, f'{state_name}_mask_short_term.tif')

    # convert all zeros mosaicked raster to a csv file [x, y, z]
    template_csv_file = os.path.join(output_directory, 'template_as_csv.csv')

    # template raster file as all zeros
    template_raster_file = os.path.join(output_directory, f"{state_name}_footprint_zero.tif")

    # output urban population file
    urban_mosaic_1990_file = os.path.join(output_directory, f"{state_name}_urban_1990_1km.tif")
    urban_mosaic_2000_file = os.path.join(output_directory, f"{state_name}_urban_2000_1km.tif")
    urban_mosaic_2010_file = os.path.join(output_directory, f"{state_name}_urban_2010_1km.tif")

    # output rural population file
    rural_mosaic_1990_file = os.path.join(output_directory, f"{state_name}_rural_1990_1km.tif")
    rural_mosaic_2000_file = os.path.join(output_directory, f"{state_name}_rural_2000_1km.tif")
    rural_mosaic_2010_file = os.path.join(output_directory, f"{state_name}_rural_2010_1km.tif")

    # output zipped file
    zipped_file = os.path.join(temp_dir, f"{state_name}.zip")

    # get a list of neighboring states including the target state
    neighboring_state_list = get_neighbors(neighbor_state_file, state_name)

    # get a list of all original mask rasters
    orig_mask_raster_list = [os.path.join(data_dir, i, "inputs", f"{i}_mask_short_term.tif") for i in neighboring_state_list]

    # get a list of all urban and rural original population rasters
    orig_urban_raster_1990_list = [os.path.join(data_dir, i, "inputs", f"{i}_urban_1990_1km.tif") for i in neighboring_state_list]
    orig_urban_raster_2000_list = [os.path.join(data_dir, i, "inputs", f"{i}_urban_2000_1km.tif") for i in neighboring_state_list]
    orig_urban_raster_2010_list = [os.path.join(data_dir, i, "inputs", f"{i}_urban_2010_1km.tif") for i in neighboring_state_list]
    orig_rural_raster_1990_list = [os.path.join(data_dir, i, "inputs", f"{i}_rural_1990_1km.tif") for i in neighboring_state_list]
    orig_rural_raster_2000_list = [os.path.join(data_dir, i, "inputs", f"{i}_rural_2000_1km.tif") for i in neighboring_state_list]
    orig_rural_raster_2010_list = [os.path.join(data_dir, i, "inputs", f"{i}_rural_2010_1km.tif") for i in neighboring_state_list]

    # fix resolution of each input raster
    new_mask_raster_list = [set_affine_resolution(i, template_directory) for i in orig_mask_raster_list]

    new_urban_raster_1990_list = [set_affine_resolution(i, template_directory) for i in orig_urban_raster_1990_list]
    new_urban_raster_2000_list = [set_affine_resolution(i, template_directory) for i in orig_urban_raster_2000_list]
    new_urban_raster_2010_list = [set_affine_resolution(i, template_directory) for i in orig_urban_raster_2010_list]

    new_rural_raster_1990_list = [set_affine_resolution(i, template_directory) for i in orig_rural_raster_1990_list]
    new_rural_raster_2000_list = [set_affine_resolution(i, template_directory) for i in orig_rural_raster_2000_list]
    new_rural_raster_2010_list = [set_affine_resolution(i, template_directory) for i in orig_rural_raster_2010_list]

    # create template raster
    create_template_raster(new_mask_raster_list, template_raster_file, set_value=0)

    # open template raster and strip out info
    with rasterio.open(template_raster_file) as trast:
        template_metadata = trast.meta.copy()
        nodata = trast.nodata

    # create template raster dataframe
    template_df = create_template_dataframe(template_raster_file)

    # create a NODATA version of the template dataframe
    template_df_nodata = template_df.copy()
    template_df_nodata['value'] = template_df_nodata['value'] + nodata

    mask_arr = mosaic_all(template_metadata, template_df_nodata, [corrected_mask_file], out_mask_raster_file)

    # create state population mosaics
    print("Building new population rasters...")
    urban_1990_arr = mosaic_all(template_metadata, template_df.copy(), new_urban_raster_1990_list,
                                urban_mosaic_1990_file)
    urban_2000_arr = mosaic_all(template_metadata, template_df.copy(), new_urban_raster_2000_list,
                                urban_mosaic_2000_file)
    urban_2010_arr = mosaic_all(template_metadata, template_df.copy(), new_urban_raster_2010_list,
                                urban_mosaic_2010_file)

    rural_1990_arr = mosaic_all(template_metadata, template_df.copy(), new_rural_raster_1990_list,
                                rural_mosaic_1990_file)
    rural_2000_arr = mosaic_all(template_metadata, template_df.copy(), new_rural_raster_2000_list,
                                rural_mosaic_2000_file)
    rural_2010_arr = mosaic_all(template_metadata, template_df.copy(), new_rural_raster_2010_list,
                                rural_mosaic_2010_file)

    # convert all zeros mosaicked raster to a csv file [x, y, z]
    print("Building new coordinate and indicies files...")
    xyz = Raster2xyz()
    xyz.translate(template_raster_file, template_csv_file)

    # read into a data frame
    df_xyz = pd.read_csv(template_csv_file)

    # keep only coordinate fields and rename to existing format
    df_xyz.drop(columns=['z'], axis=1, inplace=True)
    df_xyz.columns = ['XCoord', 'YCoord']

    # assign index to FID
    df_xyz['FID'] = df_xyz.index

    # load original coordinate file to a dataframe
    df_coords_100km = pd.read_csv(orig_coord_file)

    # merge xyz and original to get index per FID
    df_xyz['key'] = (df_xyz['XCoord'].round(4).astype(str) + '_' + df_xyz['YCoord'].round(4).astype(str))
    df_coords_100km['key'] = (
                df_coords_100km['XCoord'].round(4).astype(str) + '_' + df_coords_100km['YCoord'].round(4).astype(str))

    mdf = pd.merge(df_xyz, df_coords_100km, on='key', how='left')
    mdf.drop(columns=['key', 'XCoord_y', 'YCoord_y'], inplace=True)
    mdf.columns = ['XCoord', 'YCoord', 'FID', 'indices']
    mdf.fillna(-1, inplace=True)
    mdf['indices'] = mdf['indices'].astype(np.int)

    # open original within_indices file
    with open(orig_indices_file, 'r') as rn:
        within_indices = np.array(simplejson.load(rn))

    # create new within_indices file
    mrx = mdf.loc[mdf['indices'].isin(within_indices)].copy()
    mrx.drop(columns=['XCoord', 'YCoord'], inplace=True)
    mrx.set_index('indices', inplace=True)

    # create dictionary of {indices: FID, ...}
    d_indices_fid = mrx.to_dict()['FID']

    # generate ordered list of new indices
    new_within_indices = [d_indices_fid[i] for i in within_indices]

    # write new within_indices file
    with open(new_within_indices_file, 'w') as wifle:
        wifle.write(str(new_within_indices))

    # write new coordinate file
    mdf.drop(columns=['indices'], inplace=True)
    mdf.sort_values('FID', inplace=True)
    mdf.set_index('FID', drop=False, inplace=True)
    mdf.index.name = None
    mdf = mdf[['XCoord', 'YCoord', 'FID']].copy()

    mdf.to_csv(new_coord_file, index=False)

    # ---- TESTS ----
    print("Running tests...")

    # read in within_indices file
    with open(new_within_indices_file, 'r') as rn:
        new_indices_read = simplejson.load(rn)
        new_indices_length = len(new_indices_read)
    #     print(f"Total indices for the new data:  {new_indices_length}")

    # read in new coordinate file
    new_coords = pd.read_csv(new_coord_file)
    new_coords_shape = new_coords.shape[0]
    # print(f"Shape of new coordinate data frame:  {new_coords.shape}")   

    # read in new mask
    new_mask_1d = rasterio.open(out_mask_raster_file).read(1).flatten()
    new_mask_1d_shape = new_mask_1d.shape[0]
    # print(f"Shape of new mask array:  {new_mask_1d_shape}")

    # read in within_indices file
    with open(orig_indices_file, 'r') as rn:
        old_indices_read = simplejson.load(rn)
        old_indices_length = len(old_indices_read)
    #     print(f"Total indices for the original data:  {old_indices_length}")

    # load new file and flatten
    with rasterio.open(urban_mosaic_1990_file) as urb90:
        urb90_1d = urb90.read(1).flatten()

    # load orig file and flatten
    urban_orig_1990_file = os.path.join(data_dir, state_name, "inputs", f"{state_name}_urban_1990_1km.tif")
    with rasterio.open(urban_orig_1990_file) as urb90orig:
        urb90orig_1d = urb90orig.read(1).flatten()

    # get max value index for orig data that fall within the target state
    old_urb90_ix = np.where(urb90orig_1d == urb90orig_1d[old_indices_read].max())[0]

    if old_urb90_ix.shape[0] > 1:
        raise ValueError("More then one max value in old urban 90.")

    old_urb90_max_index = old_urb90_ix[0]
    old_urb90_max_value = urb90orig_1d[old_urb90_max_index]

    # get index at max value in old dataset
    new_urb90_ix = np.where(urb90_1d == old_urb90_max_value)[0]

    if new_urb90_ix.shape[0] > 1:
        raise ValueError("More then one max value in new urban 90.")

    new_urb90_max_index = new_urb90_ix[0]
    new_urb90_max_value = urb90_1d[new_urb90_max_index]

    # run tests
    t = test()
    t.assertEqual(new_coords_shape, new_mask_1d_shape, msg="Mask shape not equal to coords shape")
    t.assertEqual(old_indices_length, new_indices_length, msg="Original and new indicies length not equal")

    t.assertEqual(old_indices_read.index(old_urb90_max_index), new_indices_read.index(new_urb90_max_index),
                  msg="Indices max index does not match.")
    t.assertEqual(old_urb90_max_value, new_urb90_max_value, msg="Indices max value not equal.")

    # test mask file agreement
    orig_mask_file = os.path.join(data_dir, state_name, "inputs", f"{state_name}_mask_short_term.tif")
    new_mask_file = os.path.join(output_directory, f"{state_name}_mask_short_term.tif")

    rasterio_assert(orig_mask_file, new_mask_file, within_indices, new_within_indices)

    # test population raster agreement
    orig_urban_1990_file = os.path.join(data_dir, state_name, "inputs", f"{state_name}_urban_1990_1km.tif")
    orig_urban_2000_file = os.path.join(data_dir, state_name, "inputs", f"{state_name}_urban_2000_1km.tif")
    orig_urban_2010_file = os.path.join(data_dir, state_name, "inputs", f"{state_name}_urban_2010_1km.tif")
    orig_rural_1990_file = os.path.join(data_dir, state_name, "inputs", f"{state_name}_rural_1990_1km.tif")
    orig_rural_2000_file = os.path.join(data_dir, state_name, "inputs", f"{state_name}_rural_2000_1km.tif")
    orig_rural_2010_file = os.path.join(data_dir, state_name, "inputs", f"{state_name}_rural_2010_1km.tif")

    rasterio_assert(orig_urban_1990_file, urban_mosaic_1990_file, within_indices, new_within_indices)
    rasterio_assert(orig_urban_2000_file, urban_mosaic_2000_file, within_indices, new_within_indices)
    rasterio_assert(orig_urban_2010_file, urban_mosaic_2010_file, within_indices, new_within_indices)
    rasterio_assert(orig_rural_1990_file, rural_mosaic_1990_file, within_indices, new_within_indices)
    rasterio_assert(orig_rural_2000_file, rural_mosaic_2000_file, within_indices, new_within_indices)
    rasterio_assert(orig_rural_2010_file, rural_mosaic_2010_file, within_indices, new_within_indices)

    # ---- CLEAN UP ----
    print("File cleanup...")
    remove_files = ['template_as_csv.csv',
                    f'{state_name}_footprint_zero.tif']

    for f in remove_files:
        try:
            os.remove(os.path.join(output_directory, f))
        except FileNotFoundError:
            print(f"No file for:  {f}")

    # ---- COPY OVER PROJECTION FILES ----
    print("Copying files...")
    target_dir = os.path.join(data_dir, state_name, 'inputs')

    ssps = ['ssp2', 'ssp3', 'ssp5']

    for scn in ssps:
        param_file = os.path.join(target_dir, f"{state_name}_{scn}_params.csv")
        proj_file = os.path.join(target_dir, f"{state_name}_{scn}_popproj.csv")

        shutil.copy(param_file, output_directory)
        shutil.copy(proj_file, output_directory)

    print("Compressing files...")
    xf = [os.path.join(output_directory, i) for i in os.listdir(output_directory) if state_name in i]

    with ZipFile(zipped_file, 'w') as pack:
        for f in xf:
            pack.write(f)

    print("Completed.")
