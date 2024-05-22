### Reproducibility Study on Adversarial Attacks against Robust Transformer Trackers

This repository contains the codes of the TMLR 2024 "Reproducibility Study on Adversarial Attacks against Robust Transformer Trackers". Three experiemnts on tesing adversrial robustness of transformer trackers are performed and their codes are included. The dataset, trackers, and attack method links are listed below: 

# Transformer trackers:

+ TransT: https://github.com/chenxin-dlut/TransT

+ TransT-SEG: http://data.votchallenge.net/vot2022/trackers/TransT-code-2022-05-02T12_15_01.373097.zip

+ MixFormerM: http://data.votchallenge.net/vot2022/trackers/MixFormerM-code-2022-05-04T09_55_58.619853.zip

+ ROMTrack: https://github.com/dawnyc/ROMTrack 


# Adversarial attacks:

+ RTAA: https://github.com/VISION-SJTU/RTAA/tree/main

+ SPARK: https://github.com/tsingqguo/AttackTracker

+ IoU: https://github.com/VISION-SJTU/IoUattack

+ CSA: https://github.com/MasterBin-IIAU/CSA


# Datasets:

+ VOT2022: https://www.votchallenge.net/vot2022/

+ UAV123: https://cemse.kaust.edu.sa/ivul/uav123

+ GOT10k: http://got-10k.aitestunion.com/downloads

## Docker Image

We provide a Docker image that includes all the necessary packages to run the experiments. To build the docker image run:

```
docker build . -f mixformer24.base -t mixformer24
```

To mount local directory to the docker container, run:

```
nvidia-docker run -it --rm --user root --mount type=bind,source="$(pwd)",target=/mnt mixformer24:latest
```

To run the codes, first export the esstial paths and then, use the following sample:

```
conda run -n mixformer24 /bin/bash -c "vot evaluate TransT"
```
