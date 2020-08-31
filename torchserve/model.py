import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))

import torch
import pickle

from models import build_model
from utils import read_config, imread

if __name__ == "__main__":
    FILE_PATH = os.path.dirname(os.path.realpath(__file__))
    path_config = os.path.join(FILE_PATH, 'base_extraction.yml')
    path_model = "/content/drive/Shared drives/REID/HIEN/Models/OSNet_Person_Attribute_Refactor/checkpoints/0731_232453/model_best_accuracy.pth"
    path_attribute = os.path.join(FILE_PATH, '..', 'peta_attribute.pkl')

    config = read_config(path_config, False)

    use_gpu = config['n_gpu'] > 0 and torch.cuda.is_available()
    device = torch.device('cuda:0' if use_gpu else 'cpu')
    map_location = "cuda:0" if use_gpu else torch.device('cpu')

    attribute_name = pickle.load(open(path_attribute, 'rb'))

    model, _ = build_model(config, num_classes=len(attribute_name))
    
    checkpoint = torch.load(path_model, map_location=map_location)
    
    model.load_state_dict(checkpoint['state_dict'])

    # torch.save(
    #     torch.load(
    #         os.path.join(os.path.dirname(os.path.realpath(__file__)), 'model_best_accuracy.pth'), 
    #         map_location=torch.device('cpu')
    #     )['state_dict'], os.path.join(os.path.dirname(os.path.realpath(__file__)), 'eager_model.pth'))
    
    # checkpoint = torch.load(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'eager_model.pth'), map_location=torch.device('cpu'))
    
    # model = EagerModel()
    # model.load_state_dict(checkpoint)
    # torch.save(model.state_dict(), os.path.join(os.path.dirname(os.path.realpath(__file__)), 'eager_model.pth'))
    
    batch = torch.rand(64, 3, 256, 128)
    traced_script_module = torch.jit.trace(model, batch)
    traced_script_module.save('eager_model.pt')

    """
    \cp "/content/drive/Shared drives/REID/HIEN/Models/OSNet_Person_Attribute_Refactor/checkpoints/0731_232453/model_best_accuracy.pth" /content/person_attribute_recognition/torchserve/
    
    torch-model-archiver --model-name eager_model --version 1.0 --serialized-file eager_model.pt --handler handler.py --export-path model_store -f

    torchserve --start --ncs --model-store model_store --models eager_model=eager_model.mar --ts-config config.properties
    torchserve --start --ncs --model-store model_store --ts-config config.properties

"""