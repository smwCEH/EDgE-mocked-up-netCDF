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
import glob


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

    NODATA = 255

    STARTDATE = datetime.datetime.strptime('1900-01-01', '%Y-%m-%d')

    # import scipy.misc
    # print sys.version
    # print scipy.version.version
    # # See:  http://stefaanlippens.net/scipy_unscaledimsave/
    # a = 200 * scipy.ones((8,8))
    # a[0:4, 0:4] = 80
    # print a
    # print type(a)
    # print a.min(), a.max(), a.mean(), a.std()
    # rescaled = r'E:\EDgE\seasonal-forecast\images\rescaled01.png'
    # for file in glob.glob(os.path.splitext(rescaled)[0] + '.*'):
    #     # print file
    #     os.remove(file)
    # scipy.misc.imsave(rescaled, a)
    # describe_image(rescaled)
    #
    # # Prevent rescaling of the dynamic range
    # unscaled = r'E:\EDgE\seasonal-forecast\images\unscaled01.png'
    # for file in glob.glob(os.path.splitext(unscaled)[0] + '.*'):
    #     # print file
    #     os.remove(file)
    # im = scipy.misc.toimage(a, cmin=0, cmax=255)
    # im.save(unscaled)
    # describe_image(unscaled)
    # # sys.exit()

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
    # # Summarise netcdf file
    # for attribute in nc.ncattrs():
    #     print attribute, '\t', nc.getncattr(attribute)
    # for dimension in nc.dimensions:
    #     print dimension
    #     print nc.dimensions[dimension]
    # for variable in nc.variables:
    #     print variable
    #     print nc.variables[variable]
    #
    # Define netcdf variable and get array of data
    variable = nc.variables[netcdf_variable]

    time_min, time_max = 1, 10
    quintile_min, quintile_max = 1, 5
    leadtime_min, leadtime_max = 1, 6

    y_min, y_max = 0, 950
    y_size = y_max - y_min
    x_min, x_max = 0, 1000
    x_size = x_max - x_min
    # resolution = 5000.
    # y_box_max = 5167500.
    # y_size = 100
    # y_box_min = y_box_max - (y_size * resolution)
    # x_box_min = 2762500.
    # x_size = 100
    # x_box_max = x_box_min + (x_size * resolution)
    # y_variable = np.array(nc.variables['y'])
    # print(y_variable.min(), y_variable.max())
    # x_variable = np.array(nc.variables['x'])
    # print(x_variable.min(), x_variable.max())
    # y_min = int((y_variable.max() - y_box_max) / resolution)
    # y_max   = int((y_variable.max() - y_box_min) / resolution)
    # print(y_min, y_max)
    # x_min = int((x_box_min - x_variable.min()) / resolution)
    # x_max   = int((x_box_max - x_variable.min()) / resolution)
    # print(x_min, x_max)

    rgba_bands = 4
    rgba = np.zeros((rgba_bands, y_size, (x_size * leadtime_max)), dtype=np.uint8)
    # ma.set_fill_value(rgba, NODATA)
    # print(rgba)27
    print('\n\n{0:<30}:\t{1}'.format('rgba.shape', rgba.shape))
    print('{0:<30}:\t{1}'.format('rgba.dtype', rgba.dtype))
    # print('{0:<30}:\t{1}'.format('rgba.fill_value', rgba.fill_value))

    print('\n\n')
    for time in range(time_min, time_max + 1):
        print('{0}{1:<12}:\t{2}'.format('\t' * 1, 'time', time))
        days = nc.variables['time'][time - 1]
        # print days
        date = STARTDATE + datetime.timedelta(days=days)
        print('{0}{1:<12}:\t{2}'.format('\t' * 1, 'date', date.date().strftime('%Y%m%d')))

        for quintile in range(quintile_min, quintile_max + 1):
            print('{0}{1:<12}:\t{2}'.format('\t' * 2, 'quintile', quintile))
            for leadtime in range(leadtime_min, leadtime_max + 1):
                print('{0}{1:<12}:\t{2}'.format('\t' * 3, 'leadtime', leadtime))

                # Read data slice from netcdf variable for time, quintile and leadtime into a floating point numpy masked array
                float_data = ma.array(variable[time - 1,
                                               quintile - 1,
                                               leadtime - 1,
                                               y_min:y_max,
                                               x_min:x_max])
                print('{0}{1:<30}:\t{2}'.format('\t' * 4, 'float_data.dtype', float_data.dtype))
                print('{0}{1:<30}:\t{2}'.format('\t' * 4, 'float_data.min()', float_data.min()))
                print('{0}{1:<30}:\t{2}'.format('\t' * 4, 'float_data.max()', float_data.max()))
                print('{0}{1:<30}:\t{2}'.format('\t' * 4, 'float_data.mean()', float_data.mean()))
                print('{0}{1:<30}:\t{2}'.format('\t' * 4, 'float_data.std()', float_data.std()))

                # Convert floating point numpy masked array to an integer (np.uint8) numpy masked array
                integer_data = np.rint(float_data).astype(np.uint8)
                print('{0}{1:<30}:\t{2}'.format('\t' * 4, 'integer_data.dtype', integer_data.dtype))
                print('{0}{1:<30}:\t{2}'.format('\t' * 4, 'integer_data.min()', integer_data.min()))
                print('{0}{1:<30}:\t{2}'.format('\t' * 4, 'integer_data.max()', integer_data.max()))
                print('{0}{1:<30}:\t{2}'.format('\t' * 4, 'integer_data.mean()', integer_data.mean()))
                print('{0}{1:<30}:\t{2}'.format('\t' * 4, 'integer_data.std()', integer_data.std()))

                # Convert integer numpy masked array to numpy array with maked values set to NODATA value
                filled_data = integer_data.filled(fill_value=NODATA)
                print('{0}{1:<30}:\t{2}'.format('\t' * 4, 'filled_data.dtype', filled_data.dtype))
                print('{0}{1:<30}:\t{2}'.format('\t' * 4, 'filled_data.min()', filled_data.min()))
                print('{0}{1:<30}:\t{2}'.format('\t' * 4, 'filled_data.max()', filled_data.max()))
                print('{0}{1:<30}:\t{2}'.format('\t' * 4, 'filled_data.mean()', filled_data.mean()))
                print('{0}{1:<30}:\t{2}'.format('\t' * 4, 'filled_data.std()', filled_data.std()))

                # summed_data = ma.sum(scaled_data, axis=1)
                # print('\n\n{0:<30}:\t{1}'.format('summed_data.fill_value', summed_data.fill_value))
                # ma.set_fill_value(summed_data, 255)
                # print('{0:<30}:\t{1}'.format('summed_data.fill_value', summed_data.fill_value))
                # print('{0:<30}:\t{1}'.format('summed_data.min()', summed_data.min()))
                # print('{0:<30}:\t{1}'.format('summed_data.max()', summed_data.max()))
                # print(summed_data)
                # print(summed_data.mask)

                rgba_x_min = (leadtime - 1) * x_size
                rgba_x_max = (leadtime * x_size)
                # print(rgba_x_min, rgba_x_max)

                if quintile < 5:
                    # print quintile - 1
                    # print 0, y_size
                    # print rgba_x_min, rgba_x_max
                    # print rgba[quintile - 1, 0:y_size, rgba_x_min:rgba_x_max].min(),\
                    #       rgba[quintile - 1, 0:y_size, rgba_x_min:rgba_x_max].max(),\
                    #       rgba[quintile - 1, 0:y_size, rgba_x_min:rgba_x_max].mean(),\
                    #       rgba[quintile - 1, 0:y_size, rgba_x_min:rgba_x_max].std()

                    np.copyto(rgba[quintile - 1, 0:y_size, rgba_x_min:rgba_x_max],
                              filled_data[0:y_size, 0:x_size],
                              casting='same_kind')
                    # rgba[quintile - 1, 0:y_size, rgba_x_min:rgba_x_max] = 100 * np.ones((y_size, x_size))
                    # rgba[quintile - 1, 0:25, rgba_x_min:rgba_x_min + 25] = 25
                    # np.copyto(rgba[quintile - 1, 0:y_size, rgba_x_min:rgba_x_max], 100)
                    # np.copyto(rgba[quintile - 1, 0:25, rgba_x_min:rgba_x_min + 25], 25)

                    print rgba[quintile - 1, 0:y_size, rgba_x_min:rgba_x_max].min(),\
                          rgba[quintile - 1, 0:y_size, rgba_x_min:rgba_x_max].max(),\
                          rgba[quintile - 1, 0:y_size, rgba_x_min:rgba_x_max].mean(),\
                          rgba[quintile - 1, 0:y_size, rgba_x_min:rgba_x_max].std()

                del float_data, integer_data, filled_data


        print('{0}{1:<30}:\t{2}'.format('\t' * 2, 'type(rgba)', type(rgba)))
        print('{0}{1:<30}:\t{2}'.format('\t' * 2, 'rgba.dtype', rgba.dtype))
        # print('{0}{1:<30}:\t{2}'.format('\t' * 2, 'rgba.fill_value', rgba.fill_value))
        print('{0}{1:<30}:\t{2}'.format('\t' * 2, 'rgba.min()', rgba.min()))
        print('{0}{1:<30}:\t{2}'.format('\t' * 2, 'rgba.max()', rgba.max()))

        for band in range(rgba.shape[0]):
            print('{0}{1:<30}:\t{2}'.format('\t' * 2, 'rgba[{0}].min()'.format(band), rgba[band].min()))
            print('{0}{1:<30}:\t{2}'.format('\t' * 2, 'rgba[{0}].max()'.format(band), rgba[band].max()))
            print('{0}{1:<30}:\t{2}'.format('\t' * 2, 'rgba[{0}].mean()'.format(band), rgba[band].mean()))
            print('{0}{1:<30}:\t{2}'.format('\t' * 2, 'rgba[{0}].std()'.format(band), rgba[band].std()))

            histogram = np.histogram(rgba[band], bins=[-1,0,100,254,255])
            print histogram
            print histogram[0]
            print histogram[0].sum()
            unique, counts = np.unique(rgba[band], return_counts=True)
            dictionary = dict(zip(unique, counts))
            for key in sorted(dictionary):
                print '{0}:\t{1}'.format(key, dictionary[key])


        image_folder = r'E:\EDgE\seasonal-forecast\images'
        image_file = 'cc-hm-ind-{0}.png'.format(date.date().strftime('%Y%m%d'))
        image_path = os.path.join(image_folder, image_file)
        for file in glob.glob(os.path.splitext(image_path)[0] + '.*'):
            # print file
            os.remove(file)
        # imsave(image_path, rgba)
        im = scipy.misc.toimage(rgba)
        # im = scipy.misc.toimage(rgba, cmin=0, cmax=255)
        # im = toimage(rgba, low=ma.min(rgba), high=ma.max(rgba))
        im.save(image_path)

        describe_image(image_path)

    nc.close()

if __name__ == '__main__':

    np.set_printoptions(precision=4, threshold=100001, suppress=True)

    main()
