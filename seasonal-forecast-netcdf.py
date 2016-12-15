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


# import scipy
# from scipy.misc import imsave
import scipy.misc


from osgeo import gdal


def describe_image(image):
    # From: https://pcjericks.github.io/py-gdalogr-cookbook/raster_layers.html
    src_ds = gdal.Open(image)
    if src_ds is None:
        print 'Unable to open {0}'.format(image)
        sys.exit(1)
    print "[ RASTER BAND COUNT ]: ", src_ds.RasterCount
    for band in range(src_ds.RasterCount):
        band += 1
        print "[ GETTING BAND ]: ", band
        srcband = src_ds.GetRasterBand(band)
        if srcband is None:
            continue
        stats = srcband.GetStatistics(True, True)
        if stats is None:
            continue
        print "[ STATS ] =  Minimum=%.3f, Maximum=%.3f, Mean=%.3f, StdDev=%.3f" % ( \
            stats[0], stats[1], stats[2], stats[3])


def main():

    # import scipy.misc
    # print sys.version
    # print scipy.version.version
    # # See:  http://stefaanlippens.net/scipy_unscaledimsave/
    # a = 200 * scipy.ones((8,8))
    # a[0:4, 0:4] = 80
    # print a
    # scipy.misc.imsave(r'E:\EDgE\seasonal-forecast\images\rescaled01.png', a)
    # # Prevent rescaling of the dynamic range
    # im = scipy.misc.toimage(a, cmin=0, cmax=255)
    # im.save(r'E:\EDgE\seasonal-forecast\images\unscaled01.png')
    # sys.exit()

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

    time_min, time_max = 1, 1
    quintile_min, quintile_max = 1, 5
    leadtime_min, leadtime_max = 1, 6

    # y_min, y_max = 0, 950
    # y_size = y_max - y_min
    # x_min, x_max = 0, 1000
    # x_size = x_max - x_min
    resolution = 5000.
    y_box_max = 5167500.
    y_size = 100
    y_box_min = y_box_max - (y_size * resolution)
    x_box_min = 2762500.
    x_size = 100
    x_box_max = x_box_min + (x_size * resolution)
    y_variable = np.array(nc.variables['y'])
    print(y_variable.min(), y_variable.max())
    x_variable = np.array(nc.variables['x'])
    print(x_variable.min(), x_variable.max())
    y_min = int((y_variable.max() - y_box_max) / resolution)
    y_max   = int((y_variable.max() - y_box_min) / resolution)
    print(y_min, y_max)
    x_min = int((x_box_min - x_variable.min()) / resolution)
    x_max   = int((x_box_max - x_variable.min()) / resolution)
    print(x_min, x_max)

    rgba = ma.empty((4, y_size, (x_size * leadtime_max)), dtype=np.uint8)
    ma.set_fill_value(rgba, 255)
    # print(rgba)
    print('\n\n{0:<30}:\t{1}'.format('rgba.shape', rgba.shape))
    print('{0:<30}:\t{1}'.format('rgba.dtype', rgba.dtype))
    print('{0:<30}:\t{1}'.format('rgba.fill_value', rgba.fill_value))

    print('\n\n')
    for time in range(time_min, time_max + 1):
        print('{0}{1:<12}:\t{2}'.format('\t' * 1, 'time', time))
        for quintile in range(quintile_min, quintile_max + 1):
            print('{0}{1:<12}:\t{2}'.format('\t' * 2, 'quintile', quintile))
            for leadtime in range(leadtime_min, leadtime_max + 1):
                print('{0}{1:<12}:\t{2}'.format('\t' * 3, 'leadtime', leadtime))

                data = ma.array(variable[time - 1,
                                        quintile - 1,
                                        leadtime - 1,
                                        y_min:y_max,
                                        x_min:x_max])
                # print('\n\n{0:<30}:\t{1}'.format('type(data)', type(data)))
                # print('{0:<30}:\t{1}'.format('data.shape', data.shape))
                # print('{0:<30}:\t{1}'.format('data.dtype', data.dtype))
                print('{0}{1:<30}:\t{2}'.format('\t' * 4, 'data.fill_value', data.fill_value))
                print('{0}{1:<30}:\t{2}'.format('\t' * 4, 'data.min()', data.min()))
                print('{0}{1:<30}:\t{2}'.format('\t' * 4, 'data.max()', data.max()))
                # print(data)
                #
                scaled_data = np.rint(data).astype(int)
                # print('\n\n{0:<30}:\t{1}'.format('scaled_data.fill_value', scaled_data.fill_value))
                ma.set_fill_value(scaled_data, 255)
                print('{0}{1:<30}:\t{2}'.format('\t' * 4, 'scaled_data.fill_value', scaled_data.fill_value))
                print('{0}{1:<30}:\t{2}'.format('\t' * 4, 'scaled_data.min()', scaled_data.min()))
                print('{0}{1:<30}:\t{2}'.format('\t' * 4, 'scaled_data.max()', scaled_data.max()))
                # print('{0:<30}:\t{1}'.format('scaled_data.dtype', scaled_data.dtype))
                # print(scaled_data)

                # summed_data = ma.sum(scaled_data, axis=1)
                # print('\n\n{0:<30}:\t{1}'.format('summed_data.fill_value', summed_data.fill_value))
                # ma.set_fill_value(summed_data, 255)
                # print('{0:<30}:\t{1}'.format('summed_data.fill_value', summed_data.fill_value))
                # print('{0:<30}:\t{1}'.format('summed_data.min()', summed_data.min()))
                # print('{0:<30}:\t{1}'.format('summed_data.max()', summed_data.max()))
                # print(summed_data)
                # print(summed_data.mask)

                rgb_x_min = (leadtime - 1) * x_size
                rgb_x_max = (leadtime * x_size) - 1
                # print(rgb_x_min, rgb_x_max)

                # rgba[0, 0:y_size, rgb_x_min:rgb_x_max + 1] = scaled_data[0:y_size, 0:x_size]
                # rgba[1, 0:y_size, rgb_x_min:rgb_x_max + 1] = scaled_data[0:y_size, 0:x_size]
                # rgba[2, 0:y_size, rgb_x_min:rgb_x_max + 1] = scaled_data[0:y_size, 0:x_size]
                # rgba[3, 0:y_size, rgb_x_min:rgb_x_max + 1] = scaled_data[0:y_size, 0:x_size]
                if quintile < 5:
                    rgba[quintile - 1, 0:y_size, rgb_x_min:rgb_x_max + 1] = scaled_data[0:y_size, 0:x_size]
                # print(rgba)

                del data, scaled_data

        print('{0}{1:<30}:\t{2}'.format('\t' * 2, 'type(rgba)', type(rgba)))
        print('{0}{1:<30}:\t{2}'.format('\t' * 2, 'rgba.dtype', rgba.dtype))
        print('{0}{1:<30}:\t{2}'.format('\t' * 2, 'rgba.fill_value', rgba.fill_value))
        print('{0}{1:<30}:\t{2}'.format('\t' * 2, 'rgba.min()', rgba.min()))
        print('{0}{1:<30}:\t{2}'.format('\t' * 2, 'rgba.max()', rgba.max()))

        print('{0}{1:<30}:\t{2}'.format('\t' * 2, 'rgba[0].min()', rgba[0].min()))
        print('{0}{1:<30}:\t{2}'.format('\t' * 2, 'rgba[0].max()', rgba[0].max()))
        print('{0}{1:<30}:\t{2}'.format('\t' * 2, 'rgba[1].min()', rgba[1].min()))
        print('{0}{1:<30}:\t{2}'.format('\t' * 2, 'rgba[1].max()', rgba[1].max()))
        print('{0}{1:<30}:\t{2}'.format('\t' * 2, 'rgba[2].min()', rgba[2].min()))
        print('{0}{1:<30}:\t{2}'.format('\t' * 2, 'rgba[2].max()', rgba[2].max()))
        print('{0}{1:<30}:\t{2}'.format('\t' * 2, 'rgba[3].min()', rgba[3].min()))
        print('{0}{1:<30}:\t{2}'.format('\t' * 2, 'rgba[3].max()', rgba[3].max()))

        image_folder = r'E:\EDgE\seasonal-forecast\images'
        image_file = 'junk{0}.png'.format(str(time).zfill(3))
        image_path = os.path.join(image_folder, image_file)
        # imsave(image_path, rgba)
        im = scipy.misc.toimage(rgba, cmin=0, cmax=255)
        # im = toimage(rgba, low=ma.min(rgba), high=ma.max(rgba))
        im.save(image_path)

        describe_image(image_path)

    nc.close()

if __name__ == '__main__':

    np.set_printoptions(precision=4, threshold=100001, suppress=True)

    main()
