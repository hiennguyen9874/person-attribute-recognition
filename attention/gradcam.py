import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))

import torch
import argparse

import cv2
import os
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from collections import defaultdict

from utils import read_config, rmdir
from data.image import build_datasource
from models import build_model


def imread(path):
    image = Image.open(path)
    return image


class FeatureExtractor:
    r"""Class for extracting activations and
    registering gradients from targetted intermediate layers"""

    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, x):
        outputs = []
        self.gradients = []
        for name, module in self.model._modules.items():
            x = module(x)
            if name in self.target_layers:
                x.register_hook(self.save_gradient)
                outputs += [x]
        return outputs, x


class ModelOutputs:
    r"""Class for making a forward pass, and getting:
    1. The network output.
    2. Activations from intermeddiate targetted layers.
    3. Gradients from intermeddiate targetted layers."""

    def __init__(self, model, feature_module, target_layers):
        self.model = model
        self.feature_module = feature_module
        self.feature_extractor = FeatureExtractor(self.feature_module, target_layers)

    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, x):
        target_activations = []
        for name, module in self.model._modules.items():
            if module == self.feature_module:
                target_activations, x = self.feature_extractor(x)
            elif "avgpool" in name.lower():
                x = module(x)
                x = x.view(x.size(0), -1)
            else:
                x = module(x)

        return target_activations, x


class GradCam:
    def __init__(self, model, feature_module, target_layer_names, use_cuda):
        self.model = model
        self.feature_module = feature_module
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        self.extractor = ModelOutputs(
            self.model, self.feature_module, target_layer_names
        )

    def forward(self, input):
        return self.model(input)

    def __call__(self, input, index=None):
        if self.cuda:
            features, output = self.extractor(input.cuda())
        else:
            features, output = self.extractor(input)

        if index == None:
            index = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)

        self.feature_module.zero_grad()
        self.model.zero_grad()
        one_hot.backward(retain_graph=True)

        grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()

        target = features[-1]
        target = target.cpu().data.numpy()[0, :]

        weights = np.mean(grads_val, axis=(2, 3))[0, :]
        cam = np.zeros(target.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * target[i, :, :]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, input.shape[2:])
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        return cam


def main(config):
    datasource = build_datasource(
        name=config["data"]["name"], root_dir=config["data"]["data_dir"]
    )

    transform = transforms.Compose(
        [
            transforms.Resize(size=datasource.get_image_size()),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    model, _ = build_model(config["model"], num_classes=len(datasource.get_attribute()))

    cfg_trainer = config["trainer"]

    use_gpu = cfg_trainer["n_gpu"] > 0 and torch.cuda.is_available()
    device = torch.device("cuda:0" if use_gpu else "cpu")
    map_location = "cuda:0" if use_gpu else torch.device("cpu")

    print("Loading checkpoint: {} ...".format(config["resume"]))
    checkpoint = torch.load(config["resume"], map_location=map_location)

    model.load_state_dict(checkpoint["state_dict"])
    # model.eval()
    # model.to(device)

    width, height = datasource.get_image_size()
    attribute_name = datasource.get_attribute()

    grad_cam = GradCam(
        model=model,
        feature_module=model.backbone,
        target_layer_names=["7"],
        use_cuda=use_gpu,
    )

    list_image = defaultdict(list)
    data_test = datasource.get_data("test")
    random.shuffle(data_test)
    for i, (img_path, label) in enumerate(data_test):
        if i == config["num"]:
            break
        label = label.astype(bool)
        img_orig = imread(img_path)
        img = torch.unsqueeze(transform(img_orig), dim=0)
        img = img.to(device)

        output = model(img)
        output = torch.sigmoid(output)
        output = torch.squeeze(output).cpu().detach().numpy()
        output[output > 0.5] = 1
        output[output <= 0.5] = 0
        output = output.astype(bool)

        # RGB image
        img_orig = cv2.cvtColor(np.array(img_orig), cv2.COLOR_RGB2BGR)
        img_orig = cv2.resize(img_orig, (height, width))

        # list_image = [img_orig]
        # title = ''
        for i, attribute in enumerate(attribute_name):
            heatmap = grad_cam(img, i)

            # normalize the heatmap
            heatmap /= np.max(heatmap)

            # heatmaps
            am = cv2.resize(heatmap, (height, width))
            am = 255 * (am - np.min(am)) / (np.max(am) - np.min(am) + 1e-12)
            am = np.uint8(np.floor(am))
            am = cv2.applyColorMap(am, cv2.COLORMAP_JET)

            # overlapped
            overlapped = img_orig * 0.7 + am * 0.3
            overlapped[overlapped > 255] = 255
            overlapped = overlapped.astype(np.uint8)

            if output[i] == True:
                list_image[attribute].append(overlapped)
            # list_image.append(overlapped)
            # title += attribute + ': pred-' + str(output[i]) + ', label-' + str(label[i]) +'   |    '

            # mng = plt.get_current_fig_manager()
            # mng.resize(*mng.window.maxsize())

            # plt.imshow(overlapped)
            # plt.show()

        # plt.title(title)
        # img = np.concatenate(list_image, axis=1)

        # mng = plt.get_current_fig_manager()
        # mng.resize(*mng.window.maxsize())

        # plt.imshow(img)
        # plt.show()

        cfg_testing = "outputs"
        rmdir(cfg_testing)
        output_dir = os.path.join(cfg_testing, "visualize")
        os.makedirs(output_dir, exist_ok=True)
        dict_index = defaultdict(int)
        for i, (key, value) in enumerate(list_image.items()):
            path_attribute = os.path.join(output_dir, key)
            os.makedirs(path_attribute, exist_ok=True)
            dict_index[key] = 0
            for j, x in enumerate(value):
                plt.imsave(
                    os.path.join(path_attribute, str(dict_index[key]) + ".png"), x
                )
                dict_index[key] += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--config",
        default="config.json",
        type=str,
        help="config file path (default: ./config.json)",
    )
    parser.add_argument(
        "--resume", default="", type=str, help="resume file path (default: .)"
    )
    parser.add_argument("--num", default=5, type=int, help="num attribute visualize")

    args = parser.parse_args()
    config = read_config(args.config)
    config.update({"resume": args.resume})
    config.update({"num": args.num})

    main(config)
