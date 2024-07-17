# Computronium

Let's build computronium! A "universal update rule" that performs predictive
coding, much like the human cortex. Scalable in space, abstraction, and time! 
Use our open-source computronium to achieve maximal mass-energy conversion to
computation.

**Setup**: 
```bash
# make + activate a virtual environment (optional):
python3 -m venv venv
source venv/bin/activate

# install dependencies
pip3 install -r requirements.txt

# install computronium package in edit mode
pip3 install -e .

# make sure your torch is the proper version for our setup
pip3 install torch --index-url https://download.pytorch.org/whl/cu118
```

**Data**: Using Google's [Kinetics](https://research.google/pubs/the-kinetics-human-action-video-dataset/) dataset with help from the [CVDFoundatoin](https://github.com/cvdfoundation/kinetics-dataset).
```bash
cd dataset
./k400_downloader.sh 
./k400_extractor.sh
```

**Hyperparam Search Scripts**
```bash
# generate list of experiment calls in scripts/{current time}run_commands.txt
bash scripts/cube_main_hyperparam.sh

# scripts/execute_run_commands.sh [experiment list] [start index] [end index]
bash scripts/execute_run_commands.sh scripts/2024-07-13_11-42-29_UTCrun_commands.txt 0 256
```
[replace 1 10 with start & end indices you wish to run. There are 256 python calls with different hyperparams currently]

**Evaluation Scripts**
(Alex TODO)

## Tests
```bash
# run all tests: 
coverage run -m unittest discover

# get coverage report:
coverage report --include=computronium/*

# run a specific test:
coverage run -m unittest tests/test_utils.py
```
