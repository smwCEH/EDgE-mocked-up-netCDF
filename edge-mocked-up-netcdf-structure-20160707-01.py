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
import matplotlib.pyplot as plt


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


# df = pd.DataFrame(np.arange(start=1, stop=18, step=1, dtype=np.int),
#                   index=pd.date_range(start='31/12/2019', end='01/01/2100', freq='5A'),
#                   columns=['id'])
# random_middle, random_width = 1.00, 0.05
# random_min, random_max = random_middle - random_width, random_middle + random_width
# for rcp in ['RCP2.6', 'RCP4.5', 'RCP6.0', 'RCP8.5']:
#     print('RCP:\t\t{}'.format(rcp))
#     m = float(rcp[3:])
#     print('\tm:\t\t{}'.format(m))
#     df[rcp + '_equation'] = ((df['id'] * m) + 0.0) / 30.0
#     df[rcp + '_random'] = np.random.uniform(low=random_min, high=random_max, size=(17))
#     df[rcp + '_value'] = df[rcp + '_equation'] * df[rcp + '_random']
# # print(df)
# df2 = df[['RCP2.6_value', 'RCP4.5_value', 'RCP6.0_value', 'RCP8.5_value']]
# print(df2)
# df2.plot(kind='line')
# plt.show()


# Define the folder to write netCDF files in
NETCDF_FOLDER = os.path.dirname(os.path.abspath(__file__))
NETCDF_FOLDER = os.path.join(NETCDF_FOLDER, 'netcdf')
print('\n\nnetcdf_folder:\t\t\t{}'.format(NETCDF_FOLDER))
if not os.path.exists(NETCDF_FOLDER):
    os.makedirs(NETCDF_FOLDER)


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


# Define 2D array shape
TWOD_ARRAY_SHAPE = (len(ys), len(xs))
print('\n\nTWOD_ARRAY_SHAPE:\t\t{}'.format(TWOD_ARRAY_SHAPE))


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


latitude_array = np.empty(shape=TWOD_ARRAY_SHAPE)
print('\n\nlatitude_array.shape:\t\t{}'.format(latitude_array.shape))
longitude_array = np.empty(shape=TWOD_ARRAY_SHAPE)
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


# Set 2D mask based upon the first time slice of the SMI variable in the in netCDF file
in_netcdf_file_mask = in_dataset.variables['SMI'][0, :, :]


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


rows01 = np.arange(1, TWOD_ARRAY_SHAPE[0] + 1)
rows02 = np.repeat(rows01, TWOD_ARRAY_SHAPE[1])
rows03 = rows02.reshape(TWOD_ARRAY_SHAPE)
rows04 = np.flipud(rows03)
rows05 = rows04 * 0.001
# print(rows05)
print(rows05.shape)


cols01 = np.arange(1, TWOD_ARRAY_SHAPE[1] + 1)
cols02 = np.tile(cols01, TWOD_ARRAY_SHAPE[0])
cols03 = cols02.reshape(TWOD_ARRAY_SHAPE)
cols04 = cols03 * 0.001
# print(cols04)
print(cols04.shape)


# Define out netCDF file path
# out_netcdf_file_path = os.path.splitext(__file__)[0] + '.nc'
single_netcdf_file_path = 'edge-mockup-{}.nc'.format(datetime.datetime.now().strftime('%Y%m%d'))
single_netcdf_file_path = os.path.join(NETCDF_FOLDER, single_netcdf_file_path)
print('\n\nsingle_netcdf_file_path:\t{0}'.format(single_netcdf_file_path))


# Create the single netCDF file
single_netcdf_file = create_netcdf(single_netcdf_file_path)


# INDICATORS = range(1, 21, 1)
INDICATORS = range(1, 2, 1)
INDICATORS = ['indicator' + str(i).zfill(2) for i in INDICATORS]
RCP = ['RCP2_6', 'RCP4_5', 'RCP6_0', 'RCP8_5']
# RCP = ['RCP2_6']
GCM = ['HadGEM', 'ECMWF', 'CSIRO', 'ECHAMS']
# GCM = ['HadGEM']
HYDROMODEL = ['VIC', 'MHM', 'NOAA']


# Nested loops to create single and multiple output netCDF files
print('\n\nLooping for indicators, RCPs, GCMs and HydroModels...')
indicator_count = 0
for indicator in INDICATORS:
    print('{}indicator:\t{}'.format('\t' * 1,
                                    indicator))
    #
    for rcp in RCP:
        print('{}RCP:\t{}'.format('\t' * 2,
                                  rcp))
        #
        for gcm in GCM:
            print('{}GCM:\t{}'.format('\t' * 3,
                                      gcm))
            #
            for hydromodel in HYDROMODEL:
                print('{}HydroModel:\t{}'.format('\t' * 4,
                                                 hydromodel))
                #
                indicator_count += 1
                #
                netcdf_variable = '{}_{}_{}_{}'.format(rcp,
                                                       gcm,
                                                       hydromodel,
                                                       indicator)
                print('{}netCDF variable:\t{}'.format('\t' * 5,
                                                      netcdf_variable))
                #
                # Define the multiple netCDF file path
                multiple_netcdf_file_path = 'edge-mockup-{}-{}.nc'.format(datetime.datetime.now().strftime('%Y%m%d'),
                                                                          netcdf_variable)
                multiple_netcdf_file_path = os.path.join(NETCDF_FOLDER, multiple_netcdf_file_path)
                print('{}multiple_netcdf_file_path:\t{}'.format('\t' * 6,
                                                                 multiple_netcdf_file_path))
                # Create the multiple netCDF file
                multiple_netcdf_file = create_netcdf(multiple_netcdf_file_path)
                #
                # Define the variable for the multiple output netCDF files
                multiple_variable = multiple_netcdf_file.createVariable(varname=netcdf_variable,
                                                                        datatype='float32',
                                                                        dimensions=('time', 'y', 'x'),
                                                                        fill_value=NODATA,
                                                                        zlib=True)
                #
                # Define the variable for the single output netCDF file
                single_variable = single_netcdf_file.createVariable(varname=netcdf_variable,
                                                                    datatype='float32',
                                                                    dimensions=('time', 'y', 'x'),
                                                                    fill_value=NODATA,
                                                                    zlib=True)
                variable_min, variable_max = 9.99E10, -9.99E10
                for slice in range(0, time_array.size, 1):
                    # print('\t\t\tslice:\t\t\t\t{}'.format(slice))
                    random01 = np.random.randn(TWOD_ARRAY_SHAPE[0], TWOD_ARRAY_SHAPE[1])
                    day = time_steps[slice]
                    # print('\t\tday:\t\t{}'.format(day))
                    day_of_year = int(day.strftime('%j'))
                    # print('\t\tday of year:\t\t{}'.format(day_of_year))
                    annual_cycle = 10 + 15 * np.sin(2 * np.pi * (int(day.strftime('%j')) / 365.25 - 0.28))
                    # print('\t\tannual_cycle:\t\t{}'.format(annual_cycle))
                    if rcp == 'RCP2_6':
                        base = 40 + 15 * annual_cycle
                    elif rcp == 'RCP4_5':
                        base = 30 + 15 * annual_cycle
                    elif rcp == 'RCP6_0':
                        base = 20 + 15 * annual_cycle
                    elif rcp == 'RCP8_5':
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
                multiple_variable.long_name = single_variable.long_name = '{}'.format(indicator)
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
print('Looped for indicators, RCPs, GCMs and HydroModels.')


# Report total number of indicator/netCDF files created
print('indicator_count:\t\t{}'.format(indicator_count))


#  Close the single netCDF file
single_netcdf_file.close()


# Close the in netCDF file
in_dataset.close()


# Report file size of single output netCDF file
filesize(single_netcdf_file_path)


# Report file size of multiple output netCDF files
# filesize(multiple_netcdf_file_path('edge-mockup-{}_*'.format(datetime.datetime.now().strftime('%Y%m%d'))))
filesize(r'E:\EDgE\EDgE-mocked-up-netCDF\netcdf\edge-mockup-20160708-*.nc')


# Capture end_time
end_time = time.time()


# Report elapsed_time (= end_time - start_time)
print('\n\nIt took {} to execute this.'.format(hms_string(end_time - start_time)))
print('\n\nDone.\n')




