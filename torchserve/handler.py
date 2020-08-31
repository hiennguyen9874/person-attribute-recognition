import io

from PIL import Image
from torchvision import transforms

from ts.torch_handler.image_classifier import ImageClassifier

class Multi_LabelClassifier(ImageClassifier):
    image_processing = transforms.Compose([
        transforms.Resize(size=(256, 192)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    def postprocess(self, data):
        data = data
        data[data>0.5] = 1
        data[data<0.5] = 0
        return data.tolist()

