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


from mpl_toolkits.basemap import Basemap, cm
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import mpl_toolkits.basemap.pyproj as pyproj
import datetime
import netCDF4


def main():
    #
    #  Define Lambert Azimutahl Equal Area projection (EPSG:3035)
    laeaproj = pyproj.Proj('+init=EPSG:3035')
    #
    #  Define WGS84proj
    wgs84proj = pyproj.Proj('+init=EPSG:4326')
    #
    #  Convert bottom left corner and upper right corner to lat lon
    origin = pyproj.transform(laeaproj, wgs84proj, 2500000., 750000.)
    ##print 'bottom left corner = ' + str(origin)
    upper = pyproj.transform(laeaproj, wgs84proj, 7500000., 5500000.)
    ##print 'upper right corner = ' +str(upper)
    #
    #  Define netcdf files
    netcdf_folder = r'Z:\upload\edgedata\15_09_2016'
    netcdf_files = {r'ECMF_mHM_groundwater-recharge-probabilistic-quintile-distribution_monthly_1993_01_2012_05.nc': 'prob_quintile_dist'}
    #
    #  Define netcdf file path
    netcdf_path = os.path.join(netcdf_folder, netcdf_files.keys()[0])
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
    variable = nc.variables[netcdf_files[os.path.basename(netcdf_path)]]
    data = variable[0][0][0:6][0:950][0:1000]
    print(data.shape)
    #
    # Close netcdf file
    nc.close()
    #
    # Define array and create RGB colormap
    my_rgb = np.array([[192,   0,   0],
                       [237, 125,  49],
                       [255, 192,   0],
                       [ 90, 154, 213],
                       [ 68, 114, 196]])
    # print(my_rgb)
    my_rgb = my_rgb / 255.
    # print(my_rgb)
    my_cmap = matplotlib.colors.ListedColormap(my_rgb, name='my_name')
    #
    # Define matplotlib graticule
    parallels = np.arange(-50., 81., 10.)
    meridians = np.arange(-20., 81., 10.)
    #
    # Define matplotlib figure
    fig = plt.figure(figsize=(12, 8))
    #
    # Define matplotlib basemap
    m = Basemap(llcrnrlon=origin[0],
                llcrnrlat=origin[1],
                urcrnrlon=upper[0],
                urcrnrlat=upper[1],
                resolution='i',
                projection='laea',
                lon_0=10., lat_0=52.)
    #
    # Loop through rows and columns to show monthly (lead time) data as matplotlib maps
    count = 0
    rows, cols = 2, 3
    for row in range(1, rows + 1):
        for col in range(1, cols + 1):
            count += 1
            subplot = int('{0}{1}{2}'.format(rows, cols, count))
            print(subplot)
            #
            print(rows, cols, count)
            ax = fig.add_subplot(rows, cols, count)
            if count == 1:
                ny, nx = data.shape[1], data.shape[2]
                lons, lats = m.makegrid(nx, ny)  # get lat/lons of ny by nx evenly space grid.
                x, y = m(lons, lats)  # compute map proj coordinates.
            m.drawcoastlines()
            m.drawcountries()
            # m.drawparallels(parallels, labels=[1,0,0,0], fontsize=10)
            # m.drawmeridians(meridians, labels=[0,0,0,1], fontsize=10)
            m.drawparallels(parallels, labels=[0,0,0,0], fontsize=10)
            m.drawmeridians(meridians, labels=[0,0,0,0], fontsize=10)
            #
            slice = data[count-1][:][:]
            flipped = np.flipud(slice)
            #
            clevs = np.arange(flipped.min(), flipped.max(), (flipped.max() - flipped.min()) / 5.)
            print clevs.shape
            clevs = np.append(clevs, flipped.max())
            print clevs.shape
            print clevs
            #
            cs = m.contourf(x, y, flipped, clevs, cmap=my_cmap)
            #
            plt.title('lead time = {0}'.format(count - 1), fontsize=12)
    #
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(cs, cax=cbar_ax)
    fig.set_label('quintile')
    #
    plt.suptitle(netcdf_files[os.path.basename(netcdf_path)], fontsize=18)
    #
    plt.show()


if __name__ == '__main__':
    main()
