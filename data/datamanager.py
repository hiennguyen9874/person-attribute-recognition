import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))

import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data.dataloader import DataLoader

from data.image import build_datasource
from data.datasets import ImageDataset
from data.samplers import build_sampler
from data.transforms import RandomErasing

__all__ = ["DataManger_Epoch", "DataManger_Episode"]


class BaseDataManger(object):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.data_name = config["name"]

        self.datasource = build_datasource(
            name=self.data_name, root_dir=config["data_dir"]
        )

        self.dataloader = dict()

    def get_dataloader(self, phase):
        if phase not in self.datasource.get_phase():
            raise ValueError(
                "Error phase paramaster, phase in %s" % str(self.datasource.get_phase())
            )
        return self.dataloader[phase]

    def get_image_size(self):
        return self.config["image_size"][0], self.config["image_size"][1]


class DataManger_Epoch(BaseDataManger):
    def __init__(self, config, **kwargs):
        super(DataManger_Epoch, self).__init__(config)
        transform = dict()
        transform["train"] = A.Compose(
            [
                A.Resize(config["image_size"][0], config["image_size"][1]),
                A.PadIfNeeded(
                    min_height=config["image_size"][0] + 10,
                    min_width=config["image_size"][1] + 10,
                ),
                A.RandomCrop(config["image_size"][0], config["image_size"][1]),
                A.HorizontalFlip(p=0.5),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
                A.Lambda(
                    image=RandomErasing(probability=0.5, mean=[0.485, 0.456, 0.406])
                ),
            ]
        )

        transform["val"] = A.Compose(
            [
                A.Resize(config["image_size"][0], config["image_size"][1]),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )

        transform["test"] = A.Compose(
            [
                A.Resize(config["image_size"][0], config["image_size"][1]),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )

        dataset = dict()
        for _phase in self.datasource.get_phase():
            dataset[_phase] = ImageDataset(
                self.datasource.get_data(_phase), transform=transform[_phase]
            )

        self.dataloader["train"] = DataLoader(
            dataset=dataset["train"],
            batch_size=config["batch_size"],
            shuffle=config["shuffle"],
            num_workers=config["num_workers"],
            pin_memory=config["pin_memory"],
            drop_last=config["drop_last"],
        )

        self.dataloader["val"] = DataLoader(
            dataset=dataset["val"],
            batch_size=config["batch_size"],
            shuffle=False,
            num_workers=config["num_workers"],
            pin_memory=config["pin_memory"],
            drop_last=config["drop_last"],
        )

        self.dataloader["test"] = DataLoader(
            dataset["test"], batch_size=128, shuffle=False, drop_last=False
        )

    def get_batch_size(self):
        return self.config["batch_size"]


class DataManger_Episode(BaseDataManger):
    def __init__(self, config, **kwargs):
        super(DataManger_Episode, self).__init__(config)
        transform = dict()
        transform["train"] = A.Compose(
            [
                A.Resize(config["image_size"][0], config["image_size"][1]),
                A.PadIfNeeded(
                    min_height=config["image_size"][0] + 10,
                    min_width=config["image_size"][1] + 10,
                ),
                A.RandomCrop(config["image_size"][0], config["image_size"][1]),
                A.HorizontalFlip(p=0.5),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
                A.Lambda(
                    image=RandomErasing(probability=0.5, mean=[0.485, 0.456, 0.406])
                ),
            ]
        )

        transform["val"] = A.Compose(
            [
                A.Resize(config["image_size"][0], config["image_size"][1]),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )

        transform["test"] = A.Compose(
            [
                A.Resize(config["image_size"][0], config["image_size"][1]),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )

        dataset = dict()
        for _phase in self.datasource.get_phase():
            dataset[_phase] = ImageDataset(
                self.datasource.get_data(_phase), transform=transform[_phase]
            )

        sampler = dict()
        self.params_sampler = dict()
        sampler["train"], self.params_sampler["train"] = build_sampler(
            name=config["sampler"],
            config=config,
            phase="train",
            datasource=self.datasource.get_data("train"),
            weight=self.datasource.get_weight("train"),
            attribute_name=self.datasource.get_attribute(),
        )

        sampler["val"], self.params_sampler["val"] = build_sampler(
            name=config["sampler"],
            config=config,
            phase="val",
            datasource=self.datasource.get_data("val"),
            weight=self.datasource.get_weight("train"),
            attribute_name=self.datasource.get_attribute(),
        )

        self.dataloader["train"] = DataLoader(
            dataset=dataset["train"],
            batch_sampler=sampler["train"],
            num_workers=config["num_workers"],
            pin_memory=config["pin_memory"],
        )

        self.dataloader["val"] = DataLoader(
            dataset=dataset["val"],
            batch_sampler=sampler["val"],
            num_workers=config["num_workers"],
            pin_memory=config["pin_memory"],
        )

        self.dataloader["test"] = DataLoader(
            dataset["test"], batch_size=128, shuffle=False, drop_last=False
        )

    def get_batch_size(self):
        return self.config["train"]["num_attribute"] * (
            self.config["train"]["num_positive"] + self.config["train"]["num_negative"]
        )

    def get_params_sampler(self):
        return self.params_sampler


def build_datamanager(train_type, config, **kwargs):
    r"""get datamanager based type of train
    Return:
        - datamanager
        - list params of data
    """
    dict_paramsters = {"dataset": config["name"]}

    if train_type == "epoch":
        dict_paramsters.update(
            {
                "batch_size": config["batch_size"],
                "shuffle": config["shuffle"],
                "num_workers": config["num_workers"],
                "pin_memory": config["pin_memory"],
                "drop_last": config["drop_last"],
            }
        )
        return DataManger_Epoch(config, **kwargs), dict_paramsters

    elif train_type == "episode":
        datamanager = DataManger_Episode(config, **kwargs)
        dict_paramsters.update(
            {
                "sampler": config["sampler"],
            }
        )
        for x in ["train", "val"]:
            dict_paramsters.update(datamanager.params_sampler[x])
        dict_paramsters.update(
            {"num_workers": config["num_workers"], "pin_memory": config["pin_memory"]}
        )
        return datamanager, dict_paramsters
    else:
        raise KeyError("type error")
