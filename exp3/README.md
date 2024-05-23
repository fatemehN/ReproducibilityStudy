### Reproducibility Study on Adversarial Attacks against Robust Transformer Trackers
# Experiment 3: Adversarial Attack per Upper Bound

## Step1: Download the tracker

Download the ROMTrack code from the official GitHub page: https://github.com/dawnyc/ROMTrack and follow the instructions to fill the paths with your project path in 'ROMTrack/lib/test/evaluation/environment.py' to generate the 'local.py' file. 

## Step 2: Create the environment 

Follow the instructions to install the 'romtrack' environment and activate it.  

## Step 3: Download the network

Download the tracker network from the Google Drive link(https://drive.google.com/drive/folders/1Q7CpNIhWX05VU7gECnhePu3dKzTV_VoK) provided on the GitHub page. 

We tested this network in experiment 3: ROMTrack_epoch0100.pth.tar

## Step 4: Place the attack algorithms 

Copy and paste the Python files "IoU_utils.py" and "track_IoUAttack.py" to the "ROMTrack/lib/test/evaluation" directory.

## Step 5: Import the attacked version of the tracker

Comment line '11' in 'ROMTrack/tracking/test.py' and add the following line to load the attacked version of the tracker.

```
from lib.test.evaluation.tracker_IoUAttack import Tracker
```

## Step 6: Run the experiment 

The parameter '\zeta' is the 'pertrub_max' in 'IoU_utils.py'. One can change this parameter to one of the {8000, 10000, 12000} to reproduce our results by running the following command:

```
python tracking/test.py ROMTrack baseline_stage2 --dataset uav --params__model ROMTrack_epoch0100.pth.tar --params__search_area_scale 3.75
```
