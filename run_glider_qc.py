#!/usr/bin/env python

"""
Author: lgarzio on 12/7/2021
Last modified: lgarzio 12/10/2021
This is a wrapper script that imports tools to quality control RUCOOL's glider data.
"""

import argparse
import sys
import scripts

arg_parser = argparse.ArgumentParser(description="QC RUCOOL's glider data",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

arg_parser.add_argument('deployments',
                        nargs='+',
                        help='Glider deployment name(s) formatted as glider-YYYYmmddTHHMM')

arg_parser.add_argument('-m', '--mode',
                        help='Deployment dataset status',
                        choices=['rt', 'delayed'],
                        default='rt')

arg_parser.add_argument('--level',
                        choices=['sci', 'ngdac'],
                        default='sci',
                        help='Dataset type')

arg_parser.add_argument('-d', '--cdm_data_type',
                        help='Dataset type',
                        choices=['profile'],
                        default='profile')

arg_parser.add_argument('-l', '--loglevel',
                        help='Verbosity level',
                        type=str,
                        choices=['debug', 'info', 'warning', 'error', 'critical'],
                        default='info')

parsed_args = arg_parser.parse_args()

# check files that have duplicate timestamps
# scripts.check_duplicate_timestamps.main(parsed_args)

# apply QARTOD QC
# scripts.glider_qartod_qc.main(parsed_args)

# check for severely-lagged CTD profile pairs
# scripts.ctd_hysteresis_test.main(parsed_args)

# TODO summarize the QC flags

# move the files to the parent directory to be sent to ERDDAP
scripts.move_nc_files.main(parsed_args)

sys.exit()
