import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../'))

import torch
import torch.nn as nn
import torch.nn.functional as F

class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.5, device='cpu'):
        super(NTXentLoss, self).__init__()
        self.device = device
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss(reduction='sum')

    def forward(self, zi, zj):
        r""" compute normalized temperature-scaled cross entropy loss

        Args:
            zi (2d tensor (batch_size, out_dim)): [description]
            zj (2d tensor (batch_size, out_dim)): [description]

        Returns:
            loss
        """
        batch_size = zi.size(0)
        representations = torch.cat((zi, zj), dim=0)
        # representations.size() = (2*batch_size, out_dim)
        similarity = F.cosine_similarity(representations.unsqueeze(dim=1), representations.unsqueeze(dim=0), dim=-1)
        
        right = torch.diag(similarity, batch_size)
        left = torch.diag(similarity, -batch_size)
        
        positive = torch.cat((left, right)).view(2*batch_size, 1)
        negative = similarity[self.__get_negative_matrix(batch_size)].view(2*batch_size, -1)
        
        logits = torch.cat((positive, negative), dim=1)
        logits /= self.temperature
        
        labels = torch.zeros(2 * batch_size).to(self.device).long()
        loss = self.criterion(logits, labels)
        loss /= 2*batch_size
        return loss
    
    def __get_negative_matrix(self, batch_size):
        matrix = torch.ones((batch_size*2, batch_size*2), dtype=torch.bool).to(self.device)
        matrix[range(2*batch_size), range(2*batch_size)] = False
        matrix[range(batch_size), range(batch_size, 2*batch_size)] = False
        matrix[range(batch_size, 2*batch_size), range(batch_size)] = False
        return matrix

if __name__ == "__main__":
    zi = torch.rand((128, 256))
    zj = torch.rand((128, 256))
    loss = NTXentLoss()
    out = loss(zi, zj)
    print(out)