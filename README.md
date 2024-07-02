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

# make sure your torch is the proper version for our setup
pip3 install torch --index-url https://download.pytorch.org/whl/cu118
```

**Data**: Using Google's [Kinetics](https://research.google/pubs/the-kinetics-human-action-video-dataset/) dataset with help from the [CVDFoundatoin](https://github.com/cvdfoundation/kinetics-dataset).
```bash
cd dataset
./k400_downloader.sh 
./k400_extractor.sh
```
