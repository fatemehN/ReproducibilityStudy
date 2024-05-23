### Run TransT-SEG after attack for VOT2022STS evaluation

## Step 1: for TransT-SEG Attacked by CSA

1- Download the 'CSA' codes and models from its official GitHub page: https://github.com/MasterBin-IIAU/CSA

2- Copy the 'pix2pix' and 'checkpoints' containing the networks from Google Drive: https://drive.google.com/drive/folders/117GuYBQpj8Sq4yUNj7MRdyNciTCkpzXL

3- Paste them in the 'TransT/' directory of the tracker you built using 'ReproducibilityStudy/exp1/README.md'. 

4- From the TransTSEG folder on ReproducibilityStudy directory(ReproducibilityStudy/exp1/TransTSEG), copy and paste all of the python files 'ReproducibilityStudy/exp1/TransTSEG/*.py' to the (TransT/pytracking/exp1) of the tracker folder.

5- Add a new entry to the trackers.ini file in the "vot2022_workspace" directory(TransT/vot2022_workspace) as follows:
####################################

[TransT_CSA]  
label = TransT_CSA
protocol = traxpython
command =  transt_VOT2022_CSA
paths = <PATH_OF_TRANST>:<PATH_OF_TRANST>/pytracking:<PATH_OF_TRANST>/pytracking/exp1
env_PATH = <PATH_OF_PYTHON>

####################################

6- Edit the paths of the TransT_CSA entry to include all of the necessary paths as recommended on the tracker' README.md file.  The <PATH_OF_TRANST> is your path to the "TransT" folder.

7- Edit <PATH_OF_PYTHON> with your path to the 'transt' environment that you built using 'ReproducibilityStudy/exp1/README.md'. 

8- Run the evaluation:

+ Enter the VOT workspace directory (/path/to/vot2022_workspace)

+ Activate the 'transt' environment. 

+ Run:
  
    ```
    vot evaluate --workspace . TransT_CSA
    
    vot analysis --workspace .
    
    ```

## Step2: for TransT-SEG Attacked by IoU

1- From the TransTSEG folder in the ReproducibilityStudy directory(ReproducibilityStudy/code/TransTSEG), copy and paste all of the python files 'ReproducibilityStudy/exp1/TransTSEG/*.py' to the (TransT/pytracking/exp1) of the tracker folder.

2- Add a new entry to the trackers.ini file in the "vot2022_workspace" directory(TransT/vot2022_workspace) as follows:
####################################

[TransT_IoU]  
label = TransT_IoU
protocol = traxpython
command =  transt_VOT2022_IoU
paths = <PATH_OF_TRANST>:<PATH_OF_TRANST>/pytracking:<PATH_OF_TRANST>/pytracking/exp1
env_PATH = <PATH_OF_PYTHON>

####################################

3- Edit the paths of the TransT_IoU entry to include all of the necessary paths as recommended on the tracker' README.md file.  The <PATH_OF_TRANST> is your path to "TransT" folder.

4- Edit <PATH_OF_PYTHON> with your path to the 'transt' environment that you built  using 'TMLR_Supp/exp1/README.md'. 

5- Run the evaluation:

+ Enter the VOT workspace directory (/path/to/vot2022_workspace)
    
+ Activate the 'transt' environment. 
    
+ Run:

  ```
   vot evaluate --workspace . TransT_IoU
  
   vot analysis --workspace .
  
  ```

## Step3: for TransT-SEG Attacked by SPARK

1- Download the 'pysot' folder from the SPARK GitHub page: https://github.com/tsingqguo/AttackTracker/tree/main 
Then, add 'pysot' to the TransT directory as 'TransT/pysot/'.

2- Copy and paste the Python files in 'ReproducibilityStudy/exp1/SPARK/*.py' to the 'TransT/pysot/attacker/' folder. 

3- Copy and paste the Python file 'ReproducibilityStudy/exp1/TransTSEG/anchor_target.py' to the 'TransT/pysot/datasets/' directory. 

5- From the TransTSEG folder in the ReproducibilityStudy directory(ReproducibilityStudy/exp1/TransTSEG), copy and paste all of the python files 'ReproducibilityStudy/exp1/TransTSEG/*.py' to the (TransT/pytracking/exp1) of the tracker folder.

6- Add a new entry to the trackers.ini file in the "vot2022_workspace" directory(TransT/vot2022_workspace) as follows:
####################################

[TransT_SPARK]  
label = TransT_SPARK
protocol = traxpython
command =  transt_VOT2022_SPARK
paths = <PATH_OF_TRANST>:<PATH_OF_TRANST>/pytracking:<PATH_OF_TRANST>/pytracking/exp1
env_PATH = <PATH_OF_PYTHON>

####################################

3- Edit the paths of the TransT_SPARK entry to include all of the necessary paths as recommended on the tracker' README.md file.  The <PATH_OF_TRANST> is your path to the "TransT" folder.

4- Edit <PATH_OF_PYTHON> with your path to the 'transt' environment that you built  using 'TMLR_Supp/exp1/README.md'. 

5- Run the evaluation:

+ Enter the VOT workspace directory (/path/to/vot2022_workspace)
    
+ Activate the 'transt' environment. 
    
+ Run:
  
  ```
   vot evaluate --workspace . TransT_SPARK
  
   vot analysis --workspace . 
   ```


## Step5: for TransT-SEG Attacked by RTAA


5- From the TransTSEG folder in the ReproducibilityStudy directory(ReproducibilityStudy/exp1/TransTSEG), copy and paste all of the python files 'ReproducibilityStudy/exp1/TransTSEG/*.py' to the (TransT/pytracking/exp1) of the tracker folder.

6- Add a new entry to the trackers.ini file in the "vot2022_workspace" directory(TransT/vot2022_workspace) as follows:
####################################

[TransT_RTAA]  
label = TransT_RTAA
protocol = traxpython
command =  transt_VOT2022_RTAA
paths = <PATH_OF_TRANST>:<PATH_OF_TRANST>/pytracking:<PATH_OF_TRANST>/pytracking/exp1
env_PATH = <PATH_OF_PYTHON>

####################################

3- Edit the paths of the TransT_RTAA entry to include all of the necessary paths as recommended on the tracker' README.md file.  The <PATH_OF_TRANST> is your path to the "TransT" folder.

4- Edit <PATH_OF_PYTHON> with your path to the 'transt' environment that you built  using 'TMLR_Supp/exp1/README.md'. 

5- Run the evaluation:

+ Enter the VOT workspace directory (/path/to/vot2022_workspace)
    
+ Activate the 'transt' environment. 
    
+ Run:
  ```
   vot evaluate --workspace . TransT_RTAA
  
   vot analysis --workspace . 
   ```

