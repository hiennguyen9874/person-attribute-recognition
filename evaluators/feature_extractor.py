import torch
import numpy as np
from tqdm import trange, tqdm

def feature_extractor(model, data_loader, device):
    """ Extract feature from dataloader
    Args:
        model (models): 
        data_loader (Dataloader): 
        device (int): torch.device('cpu') if use_gpu == 0 else torch.device(n_gpu)
    Return:
        # TODO:
    """
    feature, label = [], []
    with torch.no_grad():
        model.to(device)
        with tqdm(total=len(data_loader)) as pbar:
            for i, batch in enumerate(data_loader):
                x, y, _ = batch
                x = x.to(device)
                e = model(x)
                feature.append(e.data.cpu())
                label.extend(y)
                pbar.update(1)
    feature = torch.cat(feature, dim=0)
    label = np.asarray(label)
    return feature, label