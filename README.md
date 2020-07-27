# Person Attribute Recognition
## Refer
- [Rethinking of Pedestrian Attribute Recognition: Realistic Datasets and A Strong Baseline](https://arxiv.org/pdf/2005.11909.pdf)
- [Bag of Tricks and A Strong Baseline for Deep Person Re-identification](https://arxiv.org/pdf/1903.07071.pdf)
- [Omni-Scale Feature Learning for Person Re-Identification](https://arxiv.org/pdf/1905.00953.pdf)
- [PyTorch Template Project by victoresque](https://github.com/victoresque/pytorch-template)
- [Torchreid](https://github.com/KaiyangZhou/deep-person-reid)

## Install
- Install [anaconda](https://docs.anaconda.com/) 
- ```conda env create -f environment.yml```
- ```conda activate reid```

## Dataset
- Automatic download and extract dataset.
- Manual download from [github.com/dangweili/pedestrian-attribute-recognition-pytorch](https://github.com/dangweili/pedestrian-attribute-recognition-pytorch)
  - PETA
    ```
    data_dir/
    |--peta/
    |  |--processed/
    |  |  |--images/
    |  |  |  |--00001.png
    |  |  |  |--00002.png
    |  |  |  |--...
    |  |  |--PETA.mat
    ```
  - PA-100K
    ```
    data_dir/
    |--pa_100k/
    |  |--processed/
    |  |  |--images/
    |  |  |  |--0000001.png
    |  |  |  |--0000002.png
    |  |  |  |--...
    |  |  |--annotation.mat
    ```
  data_dir in config file.
## Run
- ```python3 train.py --config <path/to/config_file.yml> --colab <true if run on colab else false>```

## Config
- Using config file in [config](config) folder.
- Add new config file based on [config/base.yml](config/base.yml).

## Result

### Peta dataset

| backbone          	| bn after linear 	| Head          	| Loss              	|   mA  	| Accuracy 	| Precision 	| Recall 	| F1-Score 	|
|-------------------	|:---------------:	|---------------	|-------------------	|:-----:	|:--------:	|:---------:	|:------:	|:--------:	|
| resnet50          	|     &check;     	| BNHead        	| CEL_Sigmoid       	|  83.06 	|   79.46  	|   88.48   	|  85.30 	|   86.54  	|
| resnet50          	|     &check;     	| BNHead        	| BCEWithLogitsLoss 	|  79.47 	|   76.33  	|   87.22   	|  82.38 	|   84.33  	|
| resnet50_ibn_a_nl 	|     &check;     	| BNHead        	| CEL_Sigmoid       	|  83.49 	|   79.60  	|   88.89   	|  85.14 	|   86.65  	|
| osnet             	|     &check;     	| ReductionHead 	| CEL_Sigmoid       	|  77.67 	|   73.44  	|   84.17   	|  80.60 	|   81.97  	|
| osnet             	|     &check;     	| BNHead        	| CEL_Sigmoid       	|  77.89 	|   72.57  	|   83.68   	|  79.96 	|   81.32  	|
| resnet50          	|                 	| BNHead        	| CEL_Sigmoid       	|  82.67 	|   78.61  	|   88.53   	|  84.17 	|   85.91  	|
| resnet50_ibn_a_nl 	|                 	| BNHead        	| CEL_Sigmoid       	|  82.24 	|   78.57  	|   88.48   	|  84.20 	|   85.91  	|
| osnet             	|                 	| ReductionHead 	| CEL_Sigmoid       	|  77.93 	|   73.00  	|   83.82   	|  80.65 	|   81.81  	|
| osnet             	|                 	| BNHead        	| CEL_Sigmoid       	|  77.72 	|   73.04  	|   84.65   	|  79.82 	|   81.68  	|

### PPE dataset

| backbone          	| bn after linear 	| Head          	| Loss              	|   mA  	| Accuracy 	| Precision 	| Recall 	| F1-Score 	|
|-------------------	|:---------------:	|---------------	|-------------------	|:-----:	|:--------:	|:---------:	|:------:	|:--------:	|
| resnet50          	|     &check;     	| BNHead        	| CEL_Sigmoid       	|  65.64 	|   84.63  	|   88.61   	|  88.80 	|   87.35  	|


### PA-100K

| backbone          	| bn after linear 	| Head          	| Loss              	|   mA  	| Accuracy 	| Precision 	| Recall 	| F1-Score 	|
|-------------------	|:---------------:	|---------------	|-------------------	|:-----:	|:--------:	|:---------:	|:------:	|:--------:	|
| resnet50          	|     &check;     	| BNHead        	| CEL_Sigmoid       	|        	|         	|             |       	|         	|


