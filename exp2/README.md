### Reproducibility Study on Adversarial Attacks against Robust Transformer Trackers
## Experiment 2: Adversarial Attacks per Perturbation Level

# Step 1: Download the tracker

Download the codes and models from the GitHub page of TransT tracker: https://github.com/chenxin-dlut/TransT

# Step 2: Create environment

Follow the instructions to install 'transt' environment, activate and setup the environment. 

# Step 3: Download the network

Download the tracker network. For this experiment, we used the 'transt.pth' from: https://drive.google.com/drive/folders/1GVQV1GoW-ttDJRRqaVAtLUtubtgLhWCE

# Step 4: Download the attack algorithm

Download the 'pysot' folder from SPARK Github page: https://github.com/tsingqguo/AttackTracker/tree/main 
Then, add 'pysot' to the TransT directory as 'TransT/pysot/'. 

# Step 5: Replace the SPARK algorithm with new updates to attack TransT

Copy and paste the python files in 'TMLR_supp/code/exp2/SPARK/*.py' to the 'TransT/pysot/attacker/' folder. Also, copy and past the python files is 'TMLR_supp/exp2/trackers/*.py' to the 'TransT/pysot_toolkit/trackers/' directory. 


# Step 6: Modify the paths 

Copy and paste the 'TMLR_supp/exp2/run/*.py' to the 'TransT/pysot_toolkit/'. Update the UAV123 dataset and model path in both files. 

# Step 7: Run the experiment 

- For SPARK experiment: 

a. Copy and paste the 'TMLR_supp/exp2/anchor_target.py' to the 'TransT/pysot/datasets/' directory. 

b. The perturbation level name is 'inta'. By changing the 'inta' in 'TransT/pysot/attacker/attacker_builder.py', you can apply the SPARK method with the different pertubation levels. 

c. Activate 'transt' environment.

d. Go to 'TransT/pysot_toolkit/' and run the 'test_spark.py' file. 


- For RTAA experiment: 

a. The perturbation level name is 'eps'. By changing the 'eps' in 'TransT/pysot_toolkit/trackers/tracker_RTAA.py' line 241, you can apply the RTAA method with the different pertubation levels. 

b. Activate 'transt' environment.

c. Go to 'TransT/pysot_toolkit/' and run the 'test_rtaa.py' file. 
