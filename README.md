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
- ```python3 train.py --config <path/to/config_file.yml> --colab <true if run on colab else flase>```

## Config
- Using config file in [config](config) folder.
- Add new config file based on [config/base.yml](config/base.yml).