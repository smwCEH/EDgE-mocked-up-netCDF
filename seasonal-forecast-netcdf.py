#!/usr/bin/env python3


__author__     =     'smw'
__email__      =     'smw@ceh.ac.uk'
__status__     =     'Development'


'''
Script to create images with frames for EDgE Seasonal Forecast demonstrator
Author: Simon Wright
        smw@ceh.ac.uk
        2013-12-13
'''


import os
import sys
import numpy as np
import numpy.ma as ma
import datetime
import netCDF4
from scipy.misc import imsave


def main():
    #
    #  Define netcdf files
    netcdf_folder = r'Z:\upload\edgedata\15_09_2016'
    netcdf_file = r'ECMF_mHM_groundwater-recharge-probabilistic-quintile-distribution_monthly_1993_01_2012_05.nc'
    netcdf_variable = 'prob_quintile_dist'
    #
    #  Define netcdf file path
    netcdf_path = os.path.join(netcdf_folder, netcdf_file)
    nc = netCDF4.Dataset(netcdf_path,'r')
    #
    # Summarise netcdf file
    for attribute in nc.ncattrs():
        print attribute, '\t', nc.getncattr(attribute)
    for dimension in nc.dimensions:
        print dimension
        print nc.dimensions[dimension]
    for variable in nc.variables:
        print variable
        print nc.variables[variable]
    #
    # Define netcdf variable and get array of data
    variable = nc.variables[netcdf_variable]

    # time_slice_start, time_slice_end = 0, 228
    # quintile_slice_start, quintile_slice_end = 0, 5
    # lead_time_slice_start, lead_time_slice_end = 0, 6
    # y_slice_start, y_slice_end = 0, 950
    # x_slice_start, x_slice_end = 0, 1000
    time_slice_start, time_slice_end = 0, 1
    quintile_slice_start, quintile_slice_end = 0, 5
    lead_time_slice_start, lead_time_slice_end = 0, 1

    y_slice_start, y_slice_end = 0, 950
    x_slice_start, x_slice_end = 0, 1000

    resolution = 5000.

    # y_box_max, y_box_min = 5127500., 5077500.
    y_box_max = 5167500.
    y_size = 100
    y_box_min = y_box_max - (y_size * resolution)
    # x_box_min, x_box_max = 2782500., 2832500.
    x_box_min = 2762500.
    x_size = 100
    x_box_max = x_box_min + (x_size * resolution)

    y_variable = np.array(nc.variables['y'])
    print(y_variable.min(), y_variable.max())
    x_variable = np.array(nc.variables['x'])
    print(x_variable.min(), x_variable.max())

    y_slice_start = int((y_variable.max() - y_box_max) / resolution)
    y_slice_end   = int((y_variable.max() - y_box_min) / resolution)
    print(y_slice_start, y_slice_end)
    x_slice_start = int((x_box_min - x_variable.min()) / resolution)
    x_slice_end   = int((x_box_max - x_variable.min()) / resolution)
    print(x_slice_start, x_slice_end)

    # data = variable[time_slice_start:time_slice_end][quintile_slice_start:quintile_slice_end][lead_time_slice_start:lead_time_slice_end][y_slice_start:y_slice_end][x_slice_start:x_slice_end]
    data = ma.array(variable[time_slice_start:time_slice_end,
                            quintile_slice_start:quintile_slice_end,
                            lead_time_slice_start:lead_time_slice_end,
                            y_slice_start:y_slice_end,
                            x_slice_start:x_slice_end])
    print(type(data))
    print(data.shape)
    print(data.dtype)
    print(data.fill_value)
    # print(data)

    nc.close()

    scaled_data = np.rint(data).astype(int)
    print(scaled_data.fill_value)
    ma.set_fill_value(scaled_data, 255)
    print(scaled_data.fill_value)
    print(scaled_data)
    print(scaled_data.dtype)

    summed_data = np.sum(scaled_data, axis=1)
    print(summed_data.fill_value)
    ma.set_fill_value(summed_data, 255)
    print(summed_data.fill_value)
    print(summed_data)
    print(summed_data.mask)

    rgb = np.zeros((4, y_size, x_size), dtype=np.uint8)
    # print(rgb)
    print(rgb.shape)
    print(rgb.dtype)
    rgb[0, 0:y_size, 0:x_size] = scaled_data[0, 0, 0, 0:y_size, 0:x_size]
    rgb[1, 0:y_size, 0:x_size] = scaled_data[0, 1, 0, 0:y_size, 0:x_size]
    rgb[2, 0:y_size, 0:x_size] = scaled_data[0, 2, 0, 0:y_size, 0:x_size]
    rgb[3, 0:y_size, 0:x_size] = scaled_data[0, 3, 0, 0:y_size, 0:x_size]
    print(rgb)

    image_folder = r'E:\EDgE\seasonal-forecast\images'
    image_file = 'junk01.png'
    image_path = os.path.join(image_folder, image_file)

    imsave(image_path, rgb)



if __name__ == '__main__':

    np.set_printoptions(precision=4, threshold=501, suppress=True)

    main()
