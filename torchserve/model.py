import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))

import torch
import torch.nn as nn

from models.pooling import build_pooling
from models.head import build_head
from models.backbone import build_backbone

class EagerModel(nn.Module):
    def __init__(self):
        super(EagerModel, self).__init__()
        # hard code
        num_classes=35
        backbone='resnet50'
        pretrained=True
        pooling='gem_pooling'
        pooling_size=1
        head='BNHead'
        bn_where='after'
        batch_norm_bias=True
        use_tqdm=True

        self.head_name = head
        self.num_classes = num_classes
        
        self.backbone, feature_dim = build_backbone(
            backbone, 
            pretrained=pretrained, 
            progress=use_tqdm)

        self.global_pooling = build_pooling(pooling, pooling_size)

        self.head = build_head(
            head, 
            feature_dim, 
            self.num_classes, 
            bias_freeze=not batch_norm_bias, 
            bn_where=bn_where, 
            pooling_size=pooling_size)

    def forward(self, x):
        x = self.backbone(x)
        # x.size = (batch_size, feature_dim, H, W)
        x = self.global_pooling(x)
        x = self.head(x)
        x = torch.sigmoid(x)
        return x

if __name__ == "__main__":
    torch.save(
        torch.load(
            os.path.join(os.path.dirname(os.path.realpath(__file__)), 'model_store', 'model_best_accuracy.pth'), 
            map_location=torch.device('cpu')
        )['state_dict'], os.path.join(os.path.dirname(os.path.realpath(__file__)), 'model_store', 'eager_model.pth'))
    
    
    checkpoint = torch.load(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'model_store', 'eager_model.pth'), map_location=torch.device('cpu'))
    
    model = EagerModel()
    model.load_state_dict(checkpoint)

    """
    \cp "/content/drive/Shared drives/REID/HIEN/Models/OSNet_Person_Attribute_Refactor/checkpoints/0731_232453/model_best_accuracy.pth" torchserve/model_store/
    
     torch-model-archiver --model-name EagerModel --version 1.0 --model-file model.py --serialized-file model_store/eager_model.pth --handler image_classifier
    """
