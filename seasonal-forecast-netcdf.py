#!/usr/bin/env python


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
import itertools


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


def get_combinations(array):
    combinations = []
    for length in range(len(array) + 1):
        for combination in itertools.combinations(array, length):
            if combination != ():
                combinations.append('_'.join((combination)))
    return combinations


def summarise_array(array, array_name, tabs=0):
    print('{0}{1:<30}:\t{2}'.format('\t' * tabs, array_name+'.dtype',  array.dtype))
    print('{0}{1:<30}:\t{2}'.format('\t' * tabs, array_name+'.shape',  array.shape))
    print('{0}{1:<30}:\t{2}'.format('\t' * tabs, array_name+'.min()',  array.min()))
    print('{0}{1:<30}:\t{2}'.format('\t' * tabs, array_name+'.max()',  array.max()))
    print('{0}{1:<30}:\t{2}'.format('\t' * tabs, array_name+'.mean()', array.mean()))
    print('{0}{1:<30}:\t{2}'.format('\t' * tabs, array_name+'.std()',  array.std()))


def array_histogram(array, bins_array, tabs=0):
    histogram = np.histogram(array, bins_array)
    print '\t' * tabs, histogram
    # print '\t' * tabs, histogram[0]
    print '\t' * tabs, histogram[0].sum()
    unique, counts = np.unique(array, return_counts=True)
    dictionary = dict(zip(unique, counts))
    for key in sorted(dictionary):
        print '{0}{1}:\t{2}'.format('\t' * tabs, key, dictionary[key])


def main():

    NODATA = 255

    STARTDATE = datetime.datetime.strptime('1900-01-01', '%Y-%m-%d')

    output = True

    tabs = 0

    # Define netcdf dimensions bounds
    time_min, time_max = 1, 1
    quintile_min, quintile_max = 1, 5
    leadtime_min, leadtime_max = 1, 6
    y_min, y_max = 0, 950
    y_size = y_max - y_min
    x_min, x_max = 0, 1000
    x_size = x_max - x_min

    # Define combinations of climate models
    cms = ['cm1', 'cm2', 'cm3', 'cm4']
    cm_combinations = get_combinations(cms)
    print('\n')
    print('{0}{1}:\t{2}'.format('\t' * tabs, 'cm_combinations', cm_combinations))
    print('{0}{1}:\t{2}'.format('\t' * tabs, 'len(cm_combinations)', len(cm_combinations)))

    # Define combination of hydrological models
    hms = ['hm1', 'hm2', 'hm3', 'hm4']
    hm_combinations = get_combinations(hms)
    print('\n')
    print('{0}{1}:\t{2}'.format('\t' * tabs, 'hm_combinations', hm_combinations))
    print('{0}{1}:\t{2}'.format('\t' * tabs, 'len(hm_combinations)', len(hm_combinations)))

    # Define folder for input netcdf files
    netcdf_folder = r'Z:\upload\edgedata\15_09_2016'

    # Define folder for output image files
    image_folder = r'E:\EDgE\seasonal-forecast\images'

    # Mock up time dimension as a list of floats
    time_dim = [33968.0, 33999.0, 34027.0, 34058.0, 34088.0, 34119.0, 34149.0, 34180.0, 34211.0, 34241.0, 34272.0, 34302.0, 34333.0, 34364.0, 34392.0, 34423.0, 34453.0, 34484.0, 34514.0, 34545.0, 34576.0, 34606.0, 34637.0, 34667.0, 34698.0, 34729.0, 34757.0, 34788.0, 34818.0, 34849.0, 34879.0, 34910.0, 34941.0, 34971.0, 35002.0, 35032.0, 35063.0, 35094.0, 35123.0, 35154.0, 35184.0, 35215.0, 35245.0, 35276.0, 35307.0, 35337.0, 35368.0, 35398.0, 35429.0, 35460.0, 35488.0, 35519.0, 35549.0, 35580.0, 35610.0, 35641.0, 35672.0, 35702.0, 35733.0, 35763.0, 35794.0, 35825.0, 35853.0, 35884.0, 35914.0, 35945.0, 35975.0, 36006.0, 36037.0, 36067.0, 36098.0, 36128.0, 36159.0, 36190.0, 36218.0, 36249.0, 36279.0, 36310.0, 36340.0, 36371.0, 36402.0, 36432.0, 36463.0, 36493.0, 36524.0, 36555.0, 36584.0, 36615.0, 36645.0, 36676.0, 36706.0, 36737.0, 36768.0, 36798.0, 36829.0, 36859.0, 36890.0, 36921.0, 36949.0, 36980.0, 37010.0, 37041.0, 37071.0, 37102.0, 37133.0, 37163.0, 37194.0, 37224.0, 37255.0, 37286.0, 37314.0, 37345.0, 37375.0, 37406.0, 37436.0, 37467.0, 37498.0, 37528.0, 37559.0, 37589.0, 37620.0, 37651.0, 37679.0, 37710.0, 37740.0, 37771.0, 37801.0, 37832.0, 37863.0, 37893.0, 37924.0, 37954.0, 37985.0, 38016.0, 38045.0, 38076.0, 38106.0, 38137.0, 38167.0, 38198.0, 38229.0, 38259.0, 38290.0, 38320.0, 38351.0, 38382.0, 38410.0, 38441.0, 38471.0, 38502.0, 38532.0, 38563.0, 38594.0, 38624.0, 38655.0, 38685.0, 38716.0, 38747.0, 38775.0, 38806.0, 38836.0, 38867.0, 38897.0, 38928.0, 38959.0, 38989.0, 39020.0, 39050.0, 39081.0, 39112.0, 39140.0, 39171.0, 39201.0, 39232.0, 39262.0, 39293.0, 39324.0, 39354.0, 39385.0, 39415.0, 39446.0, 39477.0, 39506.0, 39537.0, 39567.0, 39598.0, 39628.0, 39659.0, 39690.0, 39720.0, 39751.0, 39781.0, 39812.0, 39843.0, 39871.0, 39902.0, 39932.0, 39963.0, 39993.0, 40024.0, 40055.0, 40085.0, 40116.0, 40146.0, 40177.0, 40208.0, 40236.0, 40267.0, 40297.0, 40328.0, 40358.0, 40389.0, 40420.0, 40450.0, 40481.0, 40511.0, 40542.0, 40573.0, 40601.0, 40632.0, 40662.0, 40693.0, 40723.0, 40754.0, 40785.0, 40815.0, 40846.0, 40876.0]
    days = time_dim[0]

    # Loop through variable, climate model and hydrological model combinations
    print('\n')
    image_count = 0
    for var in ['var']:
        tabs += 1
        print('{0}{1}:\t{2}'.format('\t' * tabs, 'var', var))
        for cm in cm_combinations:
            tabs += 1
            print('{0}{1}:\t{2}'.format('\t' * tabs, 'cm', cm))
            for hm in hm_combinations:
                tabs += 1
                print('{0}{1}:\t{2}'.format('\t' * tabs, 'hm', hm))
                for time in range(time_min, time_max + 1):
                    tabs += 1
                    print('{0}{1}:\t{2}'.format('\t' * tabs, 'time', time))
                    if output:
                        print('{0}{1}:\t{2}'.format('\t' * tabs, 'days', days))
                    date = STARTDATE + datetime.timedelta(days=days)
                    print('{0}{1}:\t{2}'.format('\t' * tabs, 'date', date.date().strftime('%Y%m%d')))
                    tabs += 1
                    image_count += 1
                    image_file = '{0}_{1}_{2}_{3}.png'.format(var, cm, hm, date.date().strftime('%Y%m'))
                    print('{0}{1}:\t{2}'.format('\t' * tabs, 'image_file', image_file))
                    image_path = os.path.join(image_folder, image_file)
                    for file in glob.glob(os.path.splitext(image_path)[0] + '.*'):
                        # print('{0}{1:<12}:\t{2}'.format('\t' * tabs, 'file', file))
                        os.remove(file)

                    cm_list = cm.split('_')
                    # print cm_list
                    hm_list = hm.split('_')
                    # print hm_list
                    data_file_list = []
                    for cm_input in cm_list:
                        for hm_input in hm_list:
                            data_file = '{0}_{1}_{2}.nc'.format(cm_input, hm_input, var)
                            data_file_list.append(data_file)
                    tabs += 1
                    for data_file in data_file_list:
                        print('{0}{1}:\t{2}'.format('\t' * tabs, 'data_file', data_file))
                    print('{0}{1}:\t{2}'.format('\t' * tabs, 'len(data_file_list)', len(data_file_list)))
                    tabs -= 1




                    tabs -= 1

                    tabs -= 1
                tabs -= 1
            tabs -= 1
        tabs -= 1

    print('{0}{1}:\t{2}'.format('\t' * tabs, 'image_count', image_count))






    sys.exit()

    # Create netcdf4 dataset object
    netcdf_folder = r'E:\EDgE\seasonal-forecast\data'
    netcdf_file = r'ECMF_mHM_groundwater-recharge-probabilistic-quintile-distribution_monthly_1993_01_2012_05.nc'
    netcdf_path = os.path.join(netcdf_folder, netcdf_file)
    nc = netCDF4.Dataset(netcdf_path, 'r')

    # Define netcdf variable and get array of data
    variable = nc.variables['prob_quintile_dist']

    # Define rgba array
    rgba_bands = 4
    rgba = np.zeros((rgba_bands, y_size, (x_size * leadtime_max)), dtype=np.uint8)
    if output:
        print('\n\n{0:<30}:\t{1}'.format('rgba.shape', rgba.shape))
        print('{0:<30}:\t{1}'.format('rgba.dtype', rgba.dtype))

    # Mock up time dimension as a list of floats
    time_dim = [33968.0, 33999.0, 34027.0, 34058.0, 34088.0, 34119.0, 34149.0, 34180.0, 34211.0, 34241.0, 34272.0, 34302.0, 34333.0, 34364.0, 34392.0, 34423.0, 34453.0, 34484.0, 34514.0, 34545.0, 34576.0, 34606.0, 34637.0, 34667.0, 34698.0, 34729.0, 34757.0, 34788.0, 34818.0, 34849.0, 34879.0, 34910.0, 34941.0, 34971.0, 35002.0, 35032.0, 35063.0, 35094.0, 35123.0, 35154.0, 35184.0, 35215.0, 35245.0, 35276.0, 35307.0, 35337.0, 35368.0, 35398.0, 35429.0, 35460.0, 35488.0, 35519.0, 35549.0, 35580.0, 35610.0, 35641.0, 35672.0, 35702.0, 35733.0, 35763.0, 35794.0, 35825.0, 35853.0, 35884.0, 35914.0, 35945.0, 35975.0, 36006.0, 36037.0, 36067.0, 36098.0, 36128.0, 36159.0, 36190.0, 36218.0, 36249.0, 36279.0, 36310.0, 36340.0, 36371.0, 36402.0, 36432.0, 36463.0, 36493.0, 36524.0, 36555.0, 36584.0, 36615.0, 36645.0, 36676.0, 36706.0, 36737.0, 36768.0, 36798.0, 36829.0, 36859.0, 36890.0, 36921.0, 36949.0, 36980.0, 37010.0, 37041.0, 37071.0, 37102.0, 37133.0, 37163.0, 37194.0, 37224.0, 37255.0, 37286.0, 37314.0, 37345.0, 37375.0, 37406.0, 37436.0, 37467.0, 37498.0, 37528.0, 37559.0, 37589.0, 37620.0, 37651.0, 37679.0, 37710.0, 37740.0, 37771.0, 37801.0, 37832.0, 37863.0, 37893.0, 37924.0, 37954.0, 37985.0, 38016.0, 38045.0, 38076.0, 38106.0, 38137.0, 38167.0, 38198.0, 38229.0, 38259.0, 38290.0, 38320.0, 38351.0, 38382.0, 38410.0, 38441.0, 38471.0, 38502.0, 38532.0, 38563.0, 38594.0, 38624.0, 38655.0, 38685.0, 38716.0, 38747.0, 38775.0, 38806.0, 38836.0, 38867.0, 38897.0, 38928.0, 38959.0, 38989.0, 39020.0, 39050.0, 39081.0, 39112.0, 39140.0, 39171.0, 39201.0, 39232.0, 39262.0, 39293.0, 39324.0, 39354.0, 39385.0, 39415.0, 39446.0, 39477.0, 39506.0, 39537.0, 39567.0, 39598.0, 39628.0, 39659.0, 39690.0, 39720.0, 39751.0, 39781.0, 39812.0, 39843.0, 39871.0, 39902.0, 39932.0, 39963.0, 39993.0, 40024.0, 40055.0, 40085.0, 40116.0, 40146.0, 40177.0, 40208.0, 40236.0, 40267.0, 40297.0, 40328.0, 40358.0, 40389.0, 40420.0, 40450.0, 40481.0, 40511.0, 40542.0, 40573.0, 40601.0, 40632.0, 40662.0, 40693.0, 40723.0, 40754.0, 40785.0, 40815.0, 40846.0, 40876.0]
    days = time_dim[0]

    # Define image folder
    image_folder = r'E:\EDgE\seasonal-forecast\images-20161219'

    # Define tabs for output to console
    tabs = 0

    # Loop through dimensions to create images
    print('\n\n')
    for time in range(time_min, time_max + 1):
        tabs = 1
        print('{0}{1}:\t{2}'.format('\t' * tabs, 'time', time))
        if output:
            print('{0}{1}:\t{2}'.format('\t' * tabs, 'days', days))
        date = STARTDATE + datetime.timedelta(days=days)
        print('{0}{1}:\t{2}'.format('\t' * tabs, 'date', date.date().strftime('%Y%m%d')))

        for quintile in range(quintile_min, quintile_max + 1):
            tabs = 2
            print('{0}{1:<12}:\t{2}'.format('\t' * tabs, 'quintile', quintile))

            for leadtime in range(leadtime_min, leadtime_max + 1):
                tabs = 3
                print('{0}{1:<12}:\t{2}'.format('\t' * tabs, 'leadtime', leadtime))

                # Read data slice from netcdf variable for time, quintile and leadtime into a floating point numpy masked array
                float_data = ma.array(variable[time - 1,
                                               quintile - 1,
                                               leadtime - 1,
                                               y_min:y_max,
                                               x_min:x_max])
                tabs = 4
                if output:
                    summarise_array(float_data, 'float_data', tabs=tabs)

                # Convert floating point numpy masked array to an integer (np.uint8) numpy masked array
                integer_data = np.rint(float_data).astype(np.uint8)
                if output:
                    summarise_array(integer_data, 'integer_data', tabs=tabs)

                # Convert integer numpy masked array to numpy array with masked values set to NODATA value
                filled_data = integer_data.filled(fill_value=NODATA)
                if output:
                    summarise_array(filled_data, 'filled_data', tabs=tabs)

                # Calculate frame in the image by offsetting x array bounds
                rgba_x_min = (leadtime - 1) * x_size
                rgba_x_max = (leadtime * x_size)

                # For quintiles 1-4 summarise array and write to rgba 4 band array
                if quintile < 5:
                    if output:
                        print('{0}{1:<30}:\t{2}'.format('\t' * tabs, 'quintile', quintile - 1))
                        print('{0}{1:<30}:\t{2}'.format('\t' * tabs, 'rgba_x_min', rgba_x_min))
                        print('{0}{1:<30}:\t{2}'.format('\t' * tabs, 'rgba_x_max', rgba_x_max))

                    np.copyto(rgba[quintile - 1, 0:y_size, rgba_x_min:rgba_x_max], filled_data[0:y_size, 0:x_size], casting='same_kind')

                    if output:
                        array_index = 'rgba[{0},0:{1},{2}:{3}]'.format(quintile-1, y_size, rgba_x_min, rgba_x_max)
                        print('{0}{1:<30}:\t{2}'.format('\t' * tabs, array_index + '.min()', rgba[quintile - 1,0:y_size,rgba_x_min:rgba_x_max].min()))
                        print('{0}{1:<30}:\t{2}'.format('\t' * tabs, array_index + '.max()', rgba[quintile - 1,0:y_size,rgba_x_min:rgba_x_max].max()))
                        print('{0}{1:<30}:\t{2}'.format('\t' * tabs, array_index + '.mean()', rgba[quintile - 1,0:y_size,rgba_x_min:rgba_x_max].mean()))
                        print('{0}{1:<30}:\t{2}'.format('\t' * tabs, array_index + '.std()', rgba[quintile - 1,0:y_size,rgba_x_min:rgba_x_max].std()))

                # Deleet intermediate array slices
                del float_data, integer_data, filled_data

        if output:
            summarise_array(rgba, 'rgba', tabs=2)

        tabs = 2
        for band in range(rgba.shape[0]):
            if output:
                summarise_array(rgba[band], 'rgba[{0}]'.format(band), tabs=tabs)
            if output:
                array_histogram(rgba[band], [-1,0,100,254,255], tabs=tabs)

        # junk = ma.masked_where(rgba == 255, rgba)
        # print(type(junk))
        # print('{0:<30}:\t{1}'.format('junk.shape', junk.shape))
        # print('{0:<30}:\t{1}'.format('junk.min()', junk.min()))
        # print('{0:<30}:\t{1}'.format('junk.max()', junk.max()))
        # # junk[junk == 255] = 0
        # # print('{0:<30}:\t{1}'.format('junk.shape', junk.shape))
        # # print('{0:<30}:\t{1}'.format('junk.min()', junk.min()))
        # # print('{0:<30}:\t{1}'.format('junk.max()', junk.max()))
        # summed_data = ma.sum(junk, axis=0)
        # # summed_data = np.sum(np.where(rgba <= 100), axis=0)
        # print('{0:<30}:\t{1}'.format('summed_data.shape', summed_data.shape))
        # print('{0:<30}:\t{1}'.format('summed_data.min()', summed_data.min()))
        # print('{0:<30}:\t{1}'.format('summed_data.max()', summed_data.max()))
        # # print(summed_data)
        # # print summed_data[summed_data > 100]

        tabs = 1
        im = scipy.misc.toimage(rgba)
        # im = scipy.misc.toimage(rgba, cmin=0, cmax=255)
        # im = toimage(rgba, low=ma.min(rgba), high=ma.max(rgba))
        image_file = 'cc-hm-ind-{0}.png'.format(date.date().strftime('%Y%m%d'))
        image_path = os.path.join(image_folder, image_file)
        print('{0}{1}:\t{2}'.format('\t' * tabs, 'image_path', image_path))
        im.save(image_path)

        # describe_image(image_path)


    sys.exit()


    sys.exit()





    sys.exit()



if __name__ == '__main__':

    np.set_printoptions(precision=4, threshold=100001, suppress=True)

    main()
