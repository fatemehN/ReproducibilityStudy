### Reproducibility Study on Adversarial Attacks against Robust Transformer Trackers
# Experiment 1: Adversarial Attacks per Tracker Output


## Step1: Download the trackers' packages
Please download the trackers from the VOT challenge (VOT2022) website, as follows:

1- MixFormerM: http://data.votchallenge.net/vot2022/trackers/MixFormerM-code-2022-05-04T09_55_58.619853.zip
2- TransT-seg: http://data.votchallenge.net/vot2022/trackers/TransT-code-2022-05-02T12_15_01.373097.zip

## Step2: Create the environment
For each tracker follow the instructions to build a suitable environment as the instructions stated in their README.md file. 

## Step3: Download the networks 
For our experiments, we used the following networks:

1- MixFormer-M: 
1.1- Tracker network(mixformer_vit_score_imagemae.pth.tar) from https://drive.google.com/file/d/1EOZgd3HVlTmhPdsWd-zGqx4I53H4oiqf/view 
Place this network on the (MixFormerM_submit/mixformer/models)

1.2- Segmentation Network(SEcmnet_ep0440.pth.tar) from https://drive.google.com/file/d/1J0ebV0Ksye62yQOba8ymCoWFFg-MxXVy/view
Place this network on the (MixFormerM_submit/mixformer/external/AR/ltr/checkpoints/ltr/ARcm_seg/ARcm_coco_seg_only_mask_384)

2- TransT-SEG:
    TransTsegm_ep0075.pth.tar from (TransT/models/) folder of the tracker .zip file. 

## Step 4: Run the setup files 
Follow the instructions of each tracker to correct the paths and run the setup files. 

## Step 5: Tracker folders

For each tracker evaluation, follow the instructions in their folders 'TMLR_supp/exp1/*'.
