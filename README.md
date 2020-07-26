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
- Manual download from [My drive](https://drive.google.com/drive/folders/1eoiYomnR8d6SUgwL3l11jX6_x7nt6_eL?usp=sharing)
  - ```
    dataset_dir/
    |
    |--<dataset_name>/
    |   |--raw/
    |   |  |--<dataset_file.zip>
    |   |--processed/
    |   |  |--...
    ```

## Run
- ```python3 train.py --config <path/to/config_file.yml> --colab <true if run on colab else false>```

## Config
- Using config file in [config](config) folder.
- Add new config file based on [config/base.yml](config/base.yml).

## Result
| backbone          	| bn after linear 	| Head          	| Loss              	|   mA  	| Accuracy 	| Precision 	| Recall 	| F1-Score 	|
|-------------------	|:---------------:	|---------------	|-------------------	|:-----:	|:--------:	|:---------:	|:------:	|:--------:	|
| resnet50          	|     &check;     	| BNHead        	| CEL_Sigmoid       	| 83.06 	|   79.46  	|   88.48   	|  85.30 	|   86.54  	|
| resnet50          	|     &check;     	| BNHead        	| BCEWithLogitsLoss 	|       	|          	|           	|        	|          	|
| resnet50_ibn_a_nl 	|     &check;     	| BNHead        	| CEL_Sigmoid       	|  83.2 	|   79.53  	|   88.68   	|  85.22 	|   86.60  	|
| osnet             	|     &check;     	| ReductionHead 	| CEL_Sigmoid       	|       	|          	|           	|        	|          	|
| osnet             	|     &check;     	| BNHead        	| CEL_Sigmoid       	|       	|          	|           	|        	|          	|
| resnet50          	|                 	| BNHead        	| CEL_Sigmoid       	| 82.67 	|   78.61  	|   88.53   	|  84.17 	|   85.91  	|
| resnet50_ibn_a_nl 	|                 	| BNHead        	| CEL_Sigmoid       	| 82.24 	|   78.57  	|   88.48   	|  84.20 	|   85.91  	|
| osnet             	|                 	| ReductionHead 	| CEL_Sigmoid       	| 77.93 	|   73.00  	|   83.82   	|  80.65 	|   81.81  	|
| osnet             	|                 	| BNHead        	| CEL_Sigmoid       	| 77.72 	|   73.04  	|   84.65   	|  79.82 	|   81.68  	|