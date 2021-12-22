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

1. check_science_variables.py
2. check_duplicate_timestamps.py
3. glider_qartod_qc.py
4. ctd_hysteresis_test.py
5. <summarize_qc_flags - not written yet>
6. move_nc_files.py

## Acknowledgements

Development was supported in part by the [Mid-Atlantic Regional Association Coastal Ocean Observing System](https://maracoos.org/).
