import os
import glob
import shutil
import tempfile
import pkg_resources

import rasterio
import simplejson
import numpy as np
import pandas as pd
import xarray as xr

from unittest import TestCase as test
from rasterio.merge import merge


def rasterio_assert(orig_file, new_file, orig_within_indices, new_within_indices):
    """Assertion tests comparing the new versus original population rastsers."""

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


def dataframe_to_tiff(source_metadata, df, output_file, nrows, ncols):
    """Convert a data frame to a geotiff."""

    # create a 2D array in the shape of the raster
    arr = np.reshape(df['value'].astype(np.float32).values, (nrows, ncols))

    # save as raster
    with rasterio.open(output_file, 'w', **source_metadata) as dest:
        dest.write(arr, 1)

    return arr


def mosaic_tile(template_df, raster_file):
    """Mosiac individual raster file onto a template data frame."""

    # overwrite template values with those in the individual raster
    df = xr.open_rasterio(raster_file).to_dataframe(name='value')
    df.reset_index(inplace=True)

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


def mosaic_all(template_metadata, template_df, raster_list, out_file, nrows, ncols):
    """Mosaic a list of rasters onto a template raster and write the output to a file.

    :return:                  2D NumPy array

    """

    # add the incoming state population to the template
    for i in raster_list:
        template_df = mosaic_tile(template_df, i)

    # write the mosaic as a GeoTIFF
    return dataframe_to_tiff(template_metadata, template_df, out_file, nrows, ncols)


class Data:

    def __init__(self, data_dir, state_name, output_dir, target_year):

        self.state_name = state_name
        self.output_dir = os.path.join(output_dir, state_name, 'inputs')
        self.target_year = target_year
        self.temp_dir = tempfile.mkdtemp()
        self.data_dir = data_dir

        # file containing the neighboring states within 150 km from the target state
        self.neighbor_state_file = pkg_resources.resource_filename('population_gravity', 'data/neighboring_states_150km.csv')

        # make output directory if it does not exist
        make_dirs = [self.output_dir, self.temp_dir]
        for i in make_dirs:
            if not os.path.isdir(i):
                os.makedirs(i)

                # account for spaces in original state name data
        self.orig_state_name = ' '.join(self.state_name.split('_'))

        # original coordinates file
        self.orig_coord_file = os.path.join(self.data_dir, self.state_name, "inputs",
                                            f"{self.orig_state_name}_coordinates.csv")

        # new coordinates file
        self.new_coord_file = os.path.join(self.output_dir, f'{self.state_name}_coordinates.csv')

        # original within_indices file
        self.orig_indices_file = os.path.join(self.data_dir, self.state_name, "inputs",
                                              f"{self.orig_state_name}_within_indices.txt")

        # new within_indices file
        self.new_within_indices_file = os.path.join(self.output_dir, f"{self.state_name}_within_indices.txt")

        # original suitability mask file with repaired resolution
        self.corrected_mask_file = os.path.join(temp_dir, f"{self.orig_state_name}_mask_short_term.tif")

        # new suitability mask file to be mosaicked
        self.out_mask_raster_file = os.path.join(self.output_dir, f'{self.state_name}_mask_short_term.tif')

        # convert all zeros mosaicked raster to a csv file [x, y, z]
        self.template_csv_file = os.path.join(self.output_dir, 'template_as_csv.csv')

        # template raster file as all zeros
        self.template_raster_file = os.path.join(self.temp_dir, f"{self.state_name}_footprint_zero.tif")

        # original urban population file for the target state
        self.orig_urban_population_file = os.path.join(self.data_dir, self.state_name, 'inputs',
                                                       f'{self.orig_state_name}_urban_{self.target_year}_1km.tif')

        # original rural population file for the target state
        self.orig_rural_population_file = os.path.join(self.data_dir, self.state_name, 'inputs',
                                                       f'{self.orig_state_name}_rural_{self.target_year}_1km.tif')

        # output urban population file
        self.urban_mosaic_file = os.path.join(self.output_dir, f"{self.state_name}_urban_{self.target_year}_1km.tif")

        # output rural population file
        self.rural_mosaic_file = os.path.join(self.output_dir, f"{self.state_name}_rural_{self.target_year}_1km.tif")

        # output zipped file
        self.zipped_file = os.path.join(self.temp_dir, f"{self.state_name}.zip")

        # get a list of neighboring states including the target state
        self.neighboring_state_list = self.get_neighbors()

        # get a list of all original mask rasters
        self.orig_mask_raster_list = [
            os.path.join(self.data_dir, i, "inputs", f"{' '.join(i.split('_'))}_mask_short_term.tif") for i in
            self.neighboring_state_list]

        # get a list of all urban and rural original population rasters
        self.orig_urban_raster_list = [
            os.path.join(self.data_dir, i, "inputs", f"{' '.join(i.split('_'))}_urban_{self.target_year}_1km.tif") for i
            in self.neighboring_state_list]
        self.orig_rural_raster_list = [
            os.path.join(self.data_dir, i, "inputs", f"{' '.join(i.split('_'))}_rural_{self.target_year}_1km.tif") for i
            in self.neighboring_state_list]

        # fix resolution of each input raster
        self.new_mask_raster_list = [self.set_affine_resolution(i) for i in self.orig_mask_raster_list]
        self.new_urban_raster_list = [self.set_affine_resolution(i) for i in self.orig_urban_raster_list]
        self.new_rural_raster_list = [self.set_affine_resolution(i) for i in self.orig_rural_raster_list]

        # create template raster based off of a mosaic of
        self.create_template_raster(set_value=0)

        # open template raster and strip out info
        with rasterio.open(self.template_raster_file) as trast:
            self.template_metadata = trast.meta.copy()
            self.nodata = trast.nodata

        # create template raster dataframe
        self.template_df, self.nrows, self.ncols = self.create_template_dataframe()

        # create a NODATA version of the template dataframe
        self.template_df_nodata = self.template_df.copy()
        self.template_df_nodata['value'] = self.template_df_nodata['value'] + self.nodata

        # open original within_indices file
        with open(self.orig_indices_file, 'r') as rn:
            self.within_indices = np.array(simplejson.load(rn))

    def get_neighbors(self):
        """Get all neighboring states."""

        df = pd.read_csv(self.neighbor_state_file)
        states = df.loc[df['target_state'] == self.state_name]['near_state'].values

        return states

    def set_affine_resolution(self, raster_file, res=1000):
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
            out_rx = os.path.join(self.temp_dir, os.path.basename(raster_file))

            with rasterio.open(out_rx, 'w', **rx_meta) as dest:
                dest.write(rx.read())

            return out_rx

    def create_template_dataframe(self):

        # read in mosaic footprint having all 0 values
        da = xr.open_rasterio(self.template_raster_file)
        ncols = da.x.shape[0]
        nrows = da.y.shape[0]

        # convert to data frame
        df = da.to_dataframe(name='value')
        df.reset_index(inplace=True)

        # create primary key
        df['key'] = df['x'].round(4).astype(str) + '_' + df['y'].round(4).astype(str)

        return df, nrows, ncols

    def create_template_raster(self, set_value=0):
        """Create template raster mosaic set to value 0."""

        # get the metadata from the first raster in the list
        with rasterio.open(self.new_mask_raster_list[0]) as rast:
            metadata = rast.meta.copy()

        # create mosaicked raster
        src_metadata = mosaic(self.new_mask_raster_list, self.template_raster_file, metadata, set_value)


def create_indices_files(data):
    """Create new indices defining which index values from a 1D array (as flattend from its original 2D shape) correspond to grid cells
    that are within the target state.  Also create a new coordinates file that gives the X, Y coordinates of each grid cell in the new
    extent, including the NODATA grid cells.

    """

    # load original coordinate file to a dataframe
    df_coords_100km = pd.read_csv(data.orig_coord_file)

    # merge xyz and original to get index per FID
    df_coords_100km['key'] = (
                df_coords_100km['XCoord'].round(4).astype(str) + '_' + df_coords_100km['YCoord'].round(4).astype(str))

    df_coords_150km = data.template_df.copy()
    df_coords_150km.drop(columns='value', inplace=True)
    df_coords_150km['FID'] = df_coords_150km.index

    mdf = pd.merge(df_coords_150km, df_coords_100km, on='key', how='left')

    mdf.drop(columns=['key', 'XCoord', 'YCoord', 'band'], inplace=True)

    mdf.columns = ['YCoord', 'XCoord', 'FID', 'indices']
    mdf.fillna(-1, inplace=True)
    mdf['indices'] = mdf['indices'].astype(np.int64)

    # create new within_indices file
    mrx = mdf.loc[mdf['indices'].isin(data.within_indices)].copy()
    mrx.drop(columns=['XCoord', 'YCoord'], inplace=True)
    mrx.set_index('indices', inplace=True)

    # create dictionary of {indices: FID, ...}
    d_indices_fid = mrx.to_dict()['FID']

    # generate ordered list of new indices
    new_within_indices = [d_indices_fid[i] for i in data.within_indices]

    # write new within_indices file
    with open(data.new_within_indices_file, 'w') as wifle:
        wifle.write(str(new_within_indices))

    # write new coordinate file
    mdf.drop(columns=['indices'], inplace=True)
    mdf.sort_values('FID', inplace=True)
    mdf.set_index('FID', drop=False, inplace=True)
    mdf.index.name = None
    mdf = mdf[['XCoord', 'YCoord', 'FID']].copy()

    mdf.to_csv(data.new_coord_file, index=False)

    return mdf, new_within_indices


def build_new_data(data_dir, output_dir, target_year_list):
    """Full process to extend existing 100km reaching input data to 150km reaching data for the purpose
    of achieved a broader kernel distance.  Also copies over non-modified files to create a new
    directory of inputs that can be used.

    :param data_dir:            Full path to the directory containing the original input directory
    :type data_dir:             str

    :param output_dir:          Full path to the directory that will hold the newly modified inputs
    :type output_dir:           str

    :param target_year_list:    List of years as 4-digit integers to process
    :type target_year_list:     list

    """

    # get a list of all states to process
    # file containing the neighboring states within 150 km from the target state
    neighbor_state_file = pkg_resources.resource_filename('population_gravity', 'data/neighboring_states_150km.csv')
    state_list = pd.read_csv(neighbor_state_file)['target_state'].unique()

    for state_name in state_list:

        print(f"\nProcessing state:  {state_name}")

        # process each year
        for index, target_year in enumerate(target_year_list):

            print(f"Processing year:  {target_year}")

            print("Preparing data...")
            data = Data(data_dir=data_dir,
                        state_name=state_name,
                        output_dir=output_dir,
                        target_year=target_year)

            # only need to build non-population files once
            if index == 0:
                print("Building new coordinate and indicies files...")
                new_coords, new_within_indices = create_indices_files(data)

                print("Running tests to validate coordinate and indices files...")
                t = test()
                t.assertEqual(len(data.within_indices),
                              len(new_within_indices),
                              msg="Original and new indicies length not equal")

                # extend the footprint of the original mask suitability raster to include all states within the target maximum neighborhood
                #  footprint is set to NODATA for all grid cells that are not within the target state
                print("Building new suitability mask raster...")
                new_mask_arr = mosaic_all(data.template_metadata,
                                          data.template_df_nodata,
                                          [data.corrected_mask_file],
                                          data.out_mask_raster_file,
                                          data.nrows,
                                          data.ncols)

                print("Running test to validate suitability mask raster...")
                rasterio_assert(data.corrected_mask_file,
                                data.out_mask_raster_file,
                                data.within_indices,
                                new_within_indices)

                t.assertEqual(new_coords.shape[0],
                              new_mask_arr.flatten().shape[0],
                              msg="Mask shape not equal to coords shape")

            # create state population mosaics
            print("Building new population rasters...")
            urban_arr = mosaic_all(data.template_metadata,
                                   data.template_df.copy(),
                                   data.new_urban_raster_list,
                                   data.urban_mosaic_file,
                                   data.nrows,
                                   data.ncols)

            rural_arr = mosaic_all(data.template_metadata,
                                   data.template_df.copy(),
                                   data.new_rural_raster_list,
                                   data.rural_mosaic_file,
                                   data.nrows,
                                   data.ncols)

            # test population raster agreement
            print("Running tests to validate population rasters...")
            rasterio_assert(data.orig_urban_population_file, data.urban_mosaic_file, data.within_indices,
                            new_within_indices)
            rasterio_assert(data.orig_rural_population_file, data.rural_mosaic_file, data.within_indices,
                            new_within_indices)

        print("Copying additional unmodified files to the inputs directory...")
        orig_param_files = glob.glob(os.path.join(data.data_dir, data.orig_state_name, 'inputs', '*params.csv'))
        orig_proj_files = glob.glob(os.path.join(data.data_dir, data.orig_state_name, 'inputs', '*popproj.csv'))

        new_param_files = [os.path.join(data.output_dir, '_'.join(os.path.basename(i).lower().split(' '))) for i in
                           orig_param_files]
        new_proj_files = [os.path.join(data.output_dir, '_'.join(os.path.basename(i).lower().split(' '))) for i in
                          orig_proj_files]

        for idx in range(len(orig_param_files)):
            shutil.copy(orig_param_files[idx], new_param_files[idx])
            shutil.copy(orig_proj_files[idx], new_proj_files[idx])

        print("Removing the temporary directory and its contents...")
        shutil.rmtree(data.temp_dir)

        print("All tests passed.  Data created successfully.")

