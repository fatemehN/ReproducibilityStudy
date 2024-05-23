### Run MixformerM after attack for VOT2022STB evaluation

## Step 1: for MixFormerM Attacked by CSA

1- Download the 'CSA' codes and models from its official GitHub page: https://github.com/MasterBin-IIAU/CSA

2- Copy the 'pix2pix' and 'checkpoints' containing the networks from Google Drive: https://drive.google.com/drive/folders/117GuYBQpj8Sq4yUNj7MRdyNciTCkpzXL

3- Paste them in 'MixFormerM_submit/' directory. 

4- Copy and paste the python files from 'TMLR_supp/exp1/MixformerM/*.py' to the 'MixFormerM_submit/mixformer/external/AR/pytracking/exp1/'.

5- Add a new entry to trackers.ini file in "vot22_seg_mixformer_large" directory(MixFormerM_submit/mixformer/vot22_seg_mixformer_large) as follows:

######################################

[MixFormer_CSA]  
label = MixFormer_CSA
protocol = traxpython
command = mixformer_vit_large_vit_seg_class_CSA
paths = <PATH_OF_MIXFORMER>:<PATH_OF_MIXFORMER>/external/AR/pytracking/exp1:<PATH_OF_MIXFORMER>/external/AR
env_PATH = <PATH_OF_PYTHON>

#####################################

6- Edit <PATH_OF_PYTHON> with your path to the MixFormer environment that you built for this experiment from 'TMLR_supp/exp1/README.md' file.

7- The <PATH_OF_MIXFORMER> is your path to the 'mixformer' folder. Also, update the line '16' of 'siamRPNPP.py', 'GAN_utils_search_1.py' and 'GAN_utils_template_1.py' files with the  <PATH_OF_MIXFORMER>. 

8- Enter the VOT workplace directory (/path/to/vot22_seg_mixformer_large) and edit the evaluation stack to 'vot2022/stb' in the 'config.yaml'.

9- Run the MixFormer tracker attacked by CSA for VOT2022STB evaluation
    1- Enter the VOT workplace directory (/path/to/vot22_seg_mixformer_large)
    2- Activate the MixFormer environment. 
    3- Run:
        $ vot evaluate --workspace . MixFormer_CSA
        $ vot analysis --workspace . 


## Step 2: for MixFormerM Attacked by IoU 


1- Copy and paste the python files from 'TMLR_supp/exp1/MixformerM/*.py' to the 'MixFormerM_submit/mixformer/external/AR/pytracking/exp1/'.

2- Add a new entry to the trackers.ini file in "vot22_seg_mixformer_large" directory(MixFormerM_submit/mixformer/vot22_seg_mixformer_large) as follows:

######################################

[MixFormer_IoU]  
label = MixFormer_IoU
protocol = traxpython
command = mixformer_vit_large_vit_seg_class_IoU
paths = <PATH_OF_MIXFORMER>:<PATH_OF_MIXFORMER>/external/AR/pytracking/exp1:<PATH_OF_MIXFORMER>/external/AR
env_PATH = <PATH_OF_PYTHON>

#####################################

3- Edit <PATH_OF_PYTHON> with your path to the MixFormer environment that you built for this experiment from 'TMLR_supp/exp1/README.md' file.

4- Enter the VOT workplace directory (/path/to/vot22_seg_mixformer_large) and edit the evaluation stack to 'vot2022/stb' in the 'config.yaml'.

5- Run the MixFormer tracker attacked by IoU for VOT2022STB evaluation
    1- Enter the VOT workplace directory (/path/to/vot22_seg_mixformer_large)
    2- Activate the MixFormer environment. 
    3- Run:
        $ vot evaluate --workspace . MixFormer_IoU
        $ vot analysis --workspace .
        


## Note that for VOT2022STS, one should uncomment the mask prediction part from the end of the tracker class in each file, report mask instead of the bounding box, change the  stack to 'vot2022/sts' and then, run the evaluation. 
