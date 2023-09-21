#!/usr/bin/env python

"""
Author: lnazzaro 9/21/2023
Last modified: lnazzaro 9/21/2023
Check for long lag in recent data and/or gaps in older data for active deployments
"""

import argparse
import sys
from datetime import datetime, timedelta
from netCDF4 import Dataset, num2date
import pandas as pd
import numpy as np
from urllib.request import urlopen

def main(args):
    # deployment = 'ru34-20230920T1506'
    max_data_lag = args.max_lag # hours
    max_data_gap = args.max_gap # minutes
    ignore_gaps_younger_than = args.ignore_recent_gaps # hours

    for deployment in args.deployments:

        elink = f'http://slocum-data.marine.rutgers.edu/erddap/tabledap/{deployment}-profile-sci-rt'
        try:
            urlopen(elink)
        except Exception:
            print(f"{deployment} not found on ERDDAP")
            print(' ')
            continue
        glider = Dataset(elink,'r')
        t = num2date(glider['s.time'],units=glider['s.time'].units)
        latest_data_age = (datetime.utcnow() - max(t)).total_seconds()/60/60
        glider.close()

        time_gaps = pd.DataFrame({'time':t})
        time_gaps['tdiff'] = np.nan
        time_gaps.loc[1:,'tdiff'] = np.diff(time_gaps['time']).astype('timedelta64[m]').astype('float')

        print(deployment)

        lag_criteria = latest_data_age > max_data_lag
        gap_criteria = False
        if (datetime.utcnow() - t[0]).total_seconds()/60/60 > ignore_gaps_younger_than:
            gap_criteria = np.nanmax(time_gaps['tdiff'][time_gaps['time']<datetime.utcnow()-timedelta(hours=ignore_gaps_younger_than)]) > max_data_gap

        if lag_criteria or gap_criteria:
            print(f"latest sci-profile data {max(t).strftime('%Y-%m-%d %H:%M')} ({'{0:.2f}'.format(latest_data_age)} hours)")
            for i in np.where(time_gaps['tdiff'] > max_data_gap)[0]:
                tx = time_gaps['time'][i] - time_gaps['time'][i-1]
                print(f"data gap from {time_gaps['time'][i-1].strftime('%Y-%m-%d %H:%M')} to {time_gaps['time'][i].strftime('%Y-%m-%d %H:%M')} ({'{0:.2f}'.format(tx.total_seconds()/60/60)} hours)")
        else:
            print(f"lookin' good! (as far as gaps go)")
        
        print(' ')

    return

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description=main.__doc__,
                                         formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    arg_parser.add_argument('deployments',
                            nargs='+',
                            help='Glider deployment name(s) formatted as glider-YYYYmmddTHHMM')
    
    arg_parser.add_argument('-l', '--max_lag',
                            help='longest data lag (hours since latest data) to allow before triggering email',
                            default=6)
    
    arg_parser.add_argument('-g', '--max_gap',
                            help='longest data gap (minutes) to allow before triggering email',
                            default=120)
    
    arg_parser.add_argument('-r', '--ignore_recent_gaps',
                            help='amount of recent data (hours) to ignore gaps in and skip triggering email',
                            default=24)
    
    parsed_args = arg_parser.parse_args()

    sys.exit(main(parsed_args))