import os
import sys
import platform
import time
import datetime
import numpy as np
import netCDF4 as nc
import pyproj
import math
import pandas as pd
import glob


# Capture start_time
start_time = time.time()


def hms_string(sec_elapsed):
    """Function to display elapsed time

    Keyword arguments:
    sec_elapsed -- elapsed time in seconds
    """
    h = int(sec_elapsed / (60 * 60))
    m = int((sec_elapsed % (60 * 60)) / 60)
    s = sec_elapsed % 60.
    return "{}:{:>02}:{:>05.2f}".format(h, m, s)


def filesize(path):
    """Prints out total file size based upon a file path, including wildcards

    Keyword arguments:
    path -- file path, which can include wildcards
    """
    file_list = glob.glob(path)
    print('\n\n{}'.format(file_list))
    total_file_size = 0
    for file in file_list:
        total_file_size += os.stat(file).st_size
    print('total_file_size:\t\t{}'.format(total_file_size))
    print('\t\t\t\t\t\t{}'.format(filesize_format(total_file_size)))


def filesize_format(bytes, precision=2):
    """Returns a humanized string for a given amount of bytes

    Keyword arguments:
    bytes -- number of bytes to convert
    precision -- number of decimal places
    """
    bytes = int(bytes)
    if bytes is 0:
        return '0bytes'
    log = math.floor(math.log(bytes, 1024))
    return "%.*f%s" % (
        precision,
        bytes / math.pow(1024, log),
        ['B', 'kB', 'MB', 'GB', 'TB', 'PB', 'EB', 'ZB', 'YB']
        [int(log)]
    )


# Print Python version, version info, and platform architecture
print('\n\nsys.version:\t\t\t\t\t{}'.format(sys.version))
print('sys.versioninfo:\t\t\t\t{}'.format(sys.version_info))
print('platform.architecture():\t\t{}'.format(platform.architecture()))


# Define NODATA value
NODATA = -9999.0


# Seed the Numpy random number generator
np.random.seed(123)


in_netcdf_file = r'Z:\thredds\edge\E-OBS_mHM_SMI_monthly_1971_2014.nc'
print('\n\nin_netcdf_file:\t\t{}'.format(in_netcdf_file))


# Define the input netCDF file used to determine out netCDF file global attributes and spatial mask
in_dataset = nc.Dataset(in_netcdf_file, 'r')


# Describe in netCDF file
# Input netCDF file netCDF file type
print('\n\nin_dataset.file_format:\t\t{}'.format(in_dataset.file_format))
# Groups
print('\n\ndataset.groups:\t\t{}'.format(in_dataset.groups))
for group_name in in_dataset.groups.keys():
    group = in_dataset.groups[group_name]
    print('\tgroup_name:\t\t{}'.format(group_name))
# Dimensions
print('\n\nin_dataset.dimensions:\t\t{}'.format(in_dataset.dimensions))
for dim_name in in_dataset.dimensions.keys():
    dimension = in_dataset.dimensions[dim_name]
    print('\tdim_name:\t\t{}'.format(dim_name))
    print('\t\tsize:\t\t{}\n\t\tisunlimited():\t\t{}'.format(len(dimension),
                                                             dimension.isunlimited()))
# Variables
print('\n\nin_dataset.variables:\t\t{}'.format(in_dataset.variables))
for var_name in in_dataset.variables.keys():
    variable = in_dataset.variables[var_name]
    print('\tvar_name:\t\t{}'.format(var_name))
    print('\t\tdtype:\t\t{}\n\t\tdimensions():\t\t{}\n\t\tshape:\t\t{}'.format(variable.dtype,
                                                                               variable.dimensions,
                                                                               variable.shape))
# Global attributes
print('\n\nin_dataset.ncattrs():\t\t{}'.format(in_dataset.ncattrs()))
for attr_name in in_dataset.ncattrs():
    print('\tattr_name:\t\t{}'.format(attr_name))
    print('\t\tvalue:\t\t{}'.format(in_dataset.getncattr(attr_name)))
# Projection y coordinate variable
ys = in_dataset.variables['y']
print(ys)
print(type(ys))
print(ys.shape)
print(min(ys), max(ys))
# Projection x coordinate variable
xs = in_dataset.variables['x']
print(xs)
print(type(xs))
print(xs.shape)
print(min(xs), max(xs))
# Latitude variable
print('\n\n')
latitudes = in_dataset.variables['lat']
print(latitudes)
print(type(latitudes))
print(latitudes.shape)
print(latitudes[:].min(), latitudes[:].max())
# Longitude variable
print('\n\n')
longitudes = in_dataset.variables['lon']
print(longitudes)
print(type(longitudes))
print(longitudes.shape)
print(longitudes[:].min(), longitudes[:].max())


# Define time origin, units, calendar, and end
TIME_ORIGIN = datetime.datetime(year=2019,
                                month=1,
                                day=1,
                                hour=23,
                                minute=59,
                                second=59)
print('\n\nTIME_ORIGIN:\t\t{}'.format(TIME_ORIGIN))
TIME_UNITS = 'seconds since {}'.format(TIME_ORIGIN)
print('TIME_UNITS:\t\t\t{}'.format(TIME_UNITS))
TIME_CALENDAR = 'gregorian'
print('TIME_CALENDAR:\t\t{}'.format(TIME_CALENDAR))
TIME_END = datetime.datetime(year=2100,
                             month=12,
                             day=31,
                             hour=23,
                             minute=59,
                             second=59)
print('TIME_END:\t\t\t{}'.format(TIME_END))


# Define time steps as a Pandas DatetimeIndex
rng = pd.date_range(start=TIME_ORIGIN,
                    end=TIME_END,
                    freq='5A')
print(rng)
print(type(rng))


# Convert Pandas DatetimeIndex to a list of datetimes
time_steps = []
for thing in rng:
    # print(thing)
    time_steps.append(thing)
# time_steps =list(rng)
# print(time_steps)


# Convert datetime time steps array to units defined by TIME_UNITS parameter
print('time_steps:\t\t\t{}'.format(time_steps))
print('type(time_steps):\t\t{}'.format(type(time_steps)))
time_array = nc.date2num(time_steps,
                         units=TIME_UNITS,
                         calendar=TIME_CALENDAR)
print('time_array:\t\t{}'.format(time_array))
print('type(time_array):\t\t{}'.format(type(time_array)))


# Define coordinate reference system of in netCDF file data
in_proj = pyproj.Proj(init='epsg:3035')
print('\n\nin_proj.srs:\t\t{}'.format(in_proj.srs))


# Define latitude longitude coordinate reference system
out_proj = pyproj.Proj(init='epsg:4326')
print('out_proj.srs:\t\t{}'.format(out_proj.srs))


latitude_array = np.empty([len(ys), len(xs)])
print('\n\nlatitude_array.shape:\t\t{}'.format(latitude_array.shape))
longitude_array = np.empty([len(ys), len(xs)])
print('longitude_array.shape:\t\t{}'.format(longitude_array.shape))


# Convert netCDF4 projection y and x coordinate variables to numpy array allows the creation of the latitude and longitude 2D arrays to run significantly faster
ys, xs = in_dataset.variables['y'][:], in_dataset.variables['x'][:]


# Create 2D arrays of latitude and longitude based upon in netCDF file projection y and x coordinates
y_count = 0
for y in ys:
    x_count = 0
    for x in xs:
        xx, yy = pyproj.transform(in_proj,
                                  out_proj,
                                  x,
                                  y,
                                  z=None,
                                  radians=False)
        # print y_count, x_count, xx, yy
        latitude_array[y_count][x_count] = yy
        longitude_array[y_count][x_count] = xx
        x_count += 1
    y_count += 1
print('\n\nlatitude_array\n\tlatitude_array.min():\t\t{}\n\tlatitude_array.max():\t\t{}'.format(latitude_array.min(),
                                                                                                latitude_array.max()))
print('longitude_array\n\tlongitude_array.min():\t\t{}\n\tlongitude_array.max():\t\t{}'.format(longitude_array.min(),
                                                                                               longitude_array.max()))


# Define out netCDF file netCDF format
# out_netcdf_file_format = in_dataset.file_format
out_netcdf_file_format = 'NETCDF4'


def create_netcdf(out_netcdf_file_path):
    #
    # Create root group for out netCDF file
    root_group = nc.Dataset(filename=out_netcdf_file_path,
                            mode='w',
                            clobber=True,
                            diskless=False,
                            format=out_netcdf_file_format)
    #
    # Create the NetCDF file global attributes from those in the in netCDF file
    for attr_name in in_dataset.ncattrs():
        # print('\tattr_name:\t\t{}'.format(attr_name))
        attribute_value = in_dataset.getncattr(attr_name)
        # print('\t\tattribute_value:\t\t{}'.format(in_dataset.getncattr(attr_name)))
        root_group.setncattr(attr_name, attribute_value)
    root_group.modified_by = 'smw'
    root_group.modified = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    #
    #  Create out netCDF file dimensions - created in the root group
    root_group.createDimension(dimname='time',
                               size=time_array.size)
    root_group.createDimension(dimname='y',
                               size=ys.size)
    root_group.createDimension(dimname='x',
                               size=xs.size)
    #
    # Create the out netCDF file variables
    # time variable - created in the root group
    variable_time = root_group.createVariable(varname='time',
                                              datatype='double',
                                              dimensions=('time'),
                                              fill_value=NODATA,
                                              zlib=True)
    variable_time[:] = time_array
    # setattr(variable_time, 'long_name', 'time')
    # setattr(variable_time, 'units', time_units)
    # setattr(variable_time, 'calendar', time_calendar)
    variable_time.long_name = 'time'
    variable_time.units = TIME_UNITS
    variable_time.calendar = TIME_CALENDAR
    #
    # projection y coordinate variable - created in the root group
    variable_y = root_group.createVariable(varname='y',
                                           datatype='double',
                                           dimensions=('y'),
                                           # fill_value=NODATA,
                                           zlib=True)
    variable_y[:] = ys
    variable_y.standard_name = 'projection_y_coordinate'
    variable_y.long_name = 'y coordinate of projection'
    variable_y.units = 'Meter'
    #
    # projection x coordinate variable - created in the root group
    variable_x = root_group.createVariable(varname='x',
                                           datatype='double',
                                           dimensions=('x'),
                                           # fill_value=NODATA,
                                           zlib=True)
    variable_x[:] = xs
    variable_x.standard_name = 'projection_x_coordinate'
    variable_x.long_name = 'x coordinate of projection'
    variable_x.units = 'Meter'
    #
    # latitude variable - created in the root group
    variable_lat = root_group.createVariable(varname='lat',
                                             datatype='double',
                                             dimensions=('y', 'x'),
                                             # fill_value=NODATA,
                                             zlib=True)
    variable_lat[:] = latitude_array
    variable_lat.standard_name = 'latitude'
    variable_lat.long_name = 'latitude coordinate'
    variable_lat.units = 'degrees_north'
    #
    # longitude variable - created in the root group
    variable_lon = root_group.createVariable(varname='lon',
                                             datatype='double',
                                             dimensions=('y', 'x'),
                                             # fill_value=NODATA,
                                             zlib=True)
    variable_lon[:] = longitude_array
    variable_lon.standard_name = 'longitude'
    variable_lon.long_name = 'longitude coordinate'
    variable_lon.units = 'degrees_east'
    #
    # coordinate reference system variable - created in the root group
    variable_crs = root_group.createVariable(varname='lambert_azimuthal_equal_area',
                                             datatype='int',
                                             dimensions=())
    variable_crs.grid_mapping_name = 'lambert_azimuthal_equal_area'
    variable_crs.longitude_of_projection_origin = '10.0'
    variable_crs.latitude_of_projection_origin = '52.0'
    variable_crs.false_easting = '4321000.0'
    variable_crs.false_northing = '3210000.0'
    #
    return root_group


rows01 = np.arange(1, 951)
rows02 = np.repeat(rows01, 1000)
rows03 = rows02.reshape((950, 1000))
rows04 = np.flipud(rows03)
rows05 = rows04 * 0.001
# print(rows05)
# print(rows05.shape)


cols01 = np.arange(1, 1001)
cols02 = np.tile(cols01, 950)
cols03 = cols02.reshape((950, 1000))
cols04 = cols03 * 0.001
# print(cols04)
# print(cols04.shape)


# Define the folder to write netCDF files in
netcdf_folder = os.path.dirname(os.path.abspath(__file__))
netcdf_folder = os.path.join(netcdf_folder, 'netcdf')
print('\n\nnetcdf_folder:\t\t\t{}'.format(netcdf_folder))
if not os.path.exists(netcdf_folder):
    os.makedirs(netcdf_folder)


# Define out netCDF file path
# out_netcdf_file_path = os.path.splitext(__file__)[0] + '.nc'
single_netcdf_file_path = 'edge-mockup-{}.nc'.format(datetime.datetime.now().strftime('%Y%m%d'))
single_netcdf_file_path = os.path.join(netcdf_folder, single_netcdf_file_path)
print('\n\nsingle_netcdf_file_path:\t{0}'.format(single_netcdf_file_path))


# Create the single netCDF file
single_netcdf_file = create_netcdf(single_netcdf_file_path)


ENSEMBLES = 45


VARIABLES = 3


in_netcdf_file_mask = in_dataset.variables['SMI'][0, :, :]


print('\n\nLooping through ensembles...')
for ensemble in range(1, ENSEMBLES + 1, 1):
    print('\tensemble:\t\t\t{}'.format(str(ensemble).zfill(2)))
    # Define the multiple netCDF file path
    multiple_netcdf_file_path = 'edge-mockup-ensemble{}-{}.nc'.format(str(ensemble).zfill(2),
                                                                      datetime.datetime.now().strftime('%Y%m%d'))
    multiple_netcdf_file_path = os.path.join(netcdf_folder, multiple_netcdf_file_path)
    print('\t\tmultiple_netcdf_file_path:\t{0}'.format(multiple_netcdf_file_path))
    # Create the multiple netCDF file
    multiple_netcdf_file = create_netcdf(multiple_netcdf_file_path)
    #
    for variable in range(1, VARIABLES + 1, 1):
        #
        print('\t\tvariable:\t\t\t{}'.format(str(variable).zfill(2)))
        #
        multiple_variable = multiple_netcdf_file.createVariable(varname='variable{}'.format(str(variable).zfill(2)),
                                                                datatype='float32',
                                                                dimensions=('time', 'y', 'x'),
                                                                fill_value=NODATA,
                                                                zlib=True)
        # single_variable = single_netcdf_file.createVariable(varname='/ensemble{}/variable{}'.format(str(ensemble).zfill(2),
        #                                                                                             str(variable).zfill(2)),
        #                                                     datatype='float32',
        #                                                     dimensions=('time', 'y', 'x'),
        #                                                     fill_value=NODATA,
        #                                                     zlib=True)
        single_variable = single_netcdf_file.createVariable(varname='/ensemble{}_variable{}'.format(str(ensemble).zfill(2),
                                                                                                    str(variable).zfill(2)),
                                                            datatype='float32',
                                                            dimensions=('time', 'y', 'x'),
                                                            fill_value=NODATA,
                                                            zlib=True)
        variable_min, variable_max = 9.99E10, -9.99E10
        for slice in range(0, time_array.size, 1):
            # print('\t\t\tslice:\t\t\t\t{}'.format(slice))
            random01 = np.random.randn(950, 1000)
            day = time_steps[slice]
            # print('\t\tday:\t\t{}'.format(day))
            day_of_year = int(day.strftime('%j'))
            # print('\t\tday of year:\t\t{}'.format(day_of_year))
            annual_cycle = 10 + 15 * np.sin(2 * np.pi * (int(day.strftime('%j')) / 365.25 - 0.28))
            # print('\t\tannual_cycle:\t\t{}'.format(annual_cycle))
            if variable == 1:
                base = 50 + 15 * annual_cycle
            elif variable == 2:
                base = 30 + 15 * annual_cycle
            elif variable == 3:
                base = 10 + 15 * annual_cycle
            else:
                sys.exit('\n\nVariable {} not coded for!!!\n\n'.format(variable))
            mask = base + 3 * random01
            mask = mask * rows05 * cols04
            mask = np.ma.array(mask, mask=in_netcdf_file_mask.mask)
            # print('\t\t\t\tmask.shape:\t\t\t{}'.format(mask.shape))
            # print('\t\t\t\tmask.min():\t\t\t{}\n\t\t\t\tmask.max():\t\t\t{}'.format(round(float(mask.min()), 8), round(float(mask.max()), 8)))
            multiple_variable[slice] = mask
            single_variable[slice] = mask
            variable_min = min(variable_min, mask.min())
            variable_max = max(variable_max, mask.max())
            del mask
        # print('\t\tvariable_min:\t\t\t{}\n\t\tvariable_max:\t\t\t{}'.format(variable_min, variable_max))
        multiple_variable.standard_name = single_variable.standard_name = ''
        multiple_variable.long_name = single_variable.long_name = 'variable {}'.format(str(variable).zfill(2))
        multiple_variable.units = single_variable.units = '-'
        multiple_variable.coordinates = single_variable.coordinates = 'y x'
        multiple_variable.grid_mapping = single_variable.grid_mapping = 'lambert_azimuthal_equal_area'
        multiple_variable.missing_value = single_variable.missing_value = NODATA
        multiple_variable.valid_min = single_variable.valid_min = round(variable_min, 4)
        multiple_variable.valid_max = single_variable.valid_max = round(variable_max, 4)
        # ''
        # 'variable {}'.format(str(variable).zfill(2))
        # '-'
        # 'y x'
        # 'lambert_azimuthal_equal_area'
        # NODATA
        # round(variable_min, 4)
        # round(variable_max, 4)
    #
    # Close the multiple netCDF file
    multiple_netcdf_file.close()


#  Close the single netCDF file
single_netcdf_file.close()


# Close the in netCDF file
in_dataset.close()


filesize(single_netcdf_file_path)


filesize(multiple_netcdf_file_path.replace('ensemble{}'.format(str(ensemble).zfill(2)), '*'))


# file_list = glob.glob(r'E:\EDgE\Python\netcdf\edge-mockup-ensemble*-20160704.nc')
# print('\n\n{}'.format(file_list))
# total_file_size = 0
# for file in file_list:
#     total_file_size += os.stat(file).st_size
# print('total_file_size:\t\t{} bytes'.format(total_file_size))
# print('\t\t\t\t\t\t{}'.format(filesize_format(total_file_size)))


# Capture end_time
end_time = time.time()
# Report elapsed_time (= end_time - start_time)
print('\n\nIt took {} to execute this.'.format(hms_string(end_time - start_time)))
print('\n\nDone.\n')
