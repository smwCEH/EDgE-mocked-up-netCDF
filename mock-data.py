#!/usr/bin/env python


__author__     =     'smw'
__email__      =     'smw@ceh.ac.uk'
__status__     =     'Development'


'''
Script to create mock data files for EDgE Seasonal Forecast demonstrator
Author: Simon Wright
        smw@ceh.ac.uk
        2016-12-19
'''


import os
import sys
from shutil import copyfile


def main():
    # Define climate models
    cms = ['cm1', 'cm2', 'cm3', 'cm4']
    # Define hydrological models
    hms = ['hm1', 'hm2', 'hm3', 'hm4']
    # Define folder for input netcdf file
    in_netcdf_folder = r'E:\EDgE\seasonal-forecast\data'
    # Define input netcdf file
    in_netcdf_file = r'ECMF_mHM_groundwater-recharge-probabilistic-quintile-distribution_monthly_1993_01_2012_05.nc'
    in_netcdf_path = os.path.join(in_netcdf_folder, in_netcdf_file)
    # Define folder for output netcdf files
    out_netcdf_folder = r'E:\EDgE\seasonal-forecast\data\mock-data-20161219'
    # Loop through through climate models and hydrological models and copy in netcdf file to out netcdf file
    for cm in cms:
        for hm in hms:
            out_netcdf_file = in_netcdf_file.replace('ECMF', cm).replace('mHM', hm)
            out_netcdf_path = os.path.join(out_netcdf_folder, out_netcdf_file)
            if os.path.exists(out_netcdf_path):
                os.remove(out_netcdf_path)
            copyfile(in_netcdf_path, out_netcdf_path)


if __name__ == '__main__':
    main()
