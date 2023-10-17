# rugliderqc
Python tools for quality control of real-time and delayed-mode [RUCOOL glider data](https://rucool.marine.rutgers.edu/data/underwater-gliders/).

## Note: this repository is under development

## Installation

`git clone https://github.com/rucool/rugliderqc.git`

`cd rugliderqc`

`conda env create -f environment.yml`

`conda activate rugliderqc`

`pip install .`

**Requires replacing scripts in the ioos_qc package (.../envs/rugliderqc/lib/python3.9/site-packages/ioos_qc) with all files of the same name in ./ioos_qc_mods (as of 12/8/2021, only qartod.py) after the environment is created.**

## Usage

`python run_glider_qc.py glider-YYYYmmddTHHMM`

This wrapper script runs:

1. [check_science_variables.py](https://github.com/rucool/rugliderqc/blob/master/scripts/check_science_variables.py)
2. [check_duplicate_timestamps.py](https://github.com/rucool/rugliderqc/blob/master/scripts/check_duplicate_timestamps.py)
3. [glider_qartod_qc.py](https://github.com/rucool/rugliderqc/blob/master/scripts/glider_qartod_qc.py)
4. [interpolate_depth.py](https://github.com/rucool/rugliderqc/blob/master/scripts/interpolate_depth.py)
5. [ctd_hysteresis_test.py](https://github.com/rucool/rugliderqc/blob/master/scripts/ctd_hysteresis_test.py)
5. [summarize_qartod_flags.py](https://github.com/rucool/rugliderqc/blob/master/scripts/summarize_qartod_flags.py)
6. [time_shift.py](https://github.com/rucool/rugliderqc/blob/master/scripts/time_shift.py)
7. [move_nc_files.py](https://github.com/rucool/rugliderqc/blob/master/scripts/move_nc_files.py)

## Acknowledgements

Development was supported in part by the [Mid-Atlantic Regional Association Coastal Ocean Observing System](https://maracoos.org/).
