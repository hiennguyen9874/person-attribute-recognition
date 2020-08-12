import time
import torch

import sys
sys.path.append('.')

from torch.nn.utils import clip_grad_norm_

from callbacks import Tqdm
from evaluators import compute_accuracy_cuda
from trainer import Trainer

class Trainer_Episode(Trainer):
    def __init__(self, config):
        super(Trainer_Episode, self).__init__(config)
        
    def _train_epoch(self, epoch):
        r""" Training step
        """
        self.model.train()
        self.train_metrics.reset()
        if self.cfg_trainer['use_tqdm']:
            tqdm_callback = Tqdm(epoch, len(self.datamanager.get_dataloader('train')), phase='train')
        for batch_idx, (data, labels, attribute_idx) in enumerate(self.datamanager.get_dataloader('train')):
            # get time for log num iter per seconds
            if not self.cfg_trainer['use_tqdm']:
                start_time = time.time()
            # push data to device
            data, labels, attribute_idx = data.to(self.device), labels.to(self.device), attribute_idx.to(self.device)

            # zero gradient
            self.optimizer.zero_grad()

            # forward batch
            out = self.model(data)

            # calculate loss and accuracy
            loss = self.criterion(out, labels, attribute_idx)
            
            # backward parameters
            loss.backward()

            # Clips gradient norm of an iterable of parameters.
            if self.config['clip_grad_norm_']['enable']:
                clip_grad_norm_(
                    parameters=self.model.parameters(),
                    max_norm=self.config['clip_grad_norm_']['max_norm'])

            # optimize
            self.optimizer.step()
            
            # calculate instance-based accuracy
            preds = torch.sigmoid(out)
            
            mean_accuracy, accuracy, f1_score = compute_accuracy_cuda(labels, preds)

            # update loss and accuracy in MetricTracker
            self.train_metrics.update('loss', loss.item())
            self.train_metrics.update('mA', mean_accuracy)
            self.train_metrics.update('accuracy', accuracy)
            self.train_metrics.update('f1_score', f1_score)

            # update process
            if self.cfg_trainer['use_tqdm']:
                tqdm_callback.on_batch_end({
                    'loss': loss.item(),
                    'mA': mean_accuracy,
                    'accuracy': accuracy,
                    'f1-score': f1_score})
            else:
                end_time = time.time()
                if (batch_idx+1) % self.log_step[0] == 0 or (batch_idx+1) == len(self.datamanager.get_dataloader('train')):
                    self.logger.info('Train Epoch: {} {}/{} {:.1f}batch/s Loss: {:.4f} mA: {:.4f} Acc: {:.4f} F1-score: {:.4f}'.format(
                        epoch,
                        batch_idx+1,
                        len(self.datamanager.get_dataloader('train')),
                        1/(end_time-start_time),
                        loss.item(),
                        mean_accuracy,
                        accuracy,
                        f1_score))
        if self.cfg_trainer['use_tqdm']:
            tqdm_callback.on_epoch_end()
        return self.train_metrics.result()

    def _valid_epoch(self, epoch):
        r""" Validation step
        """
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            if self.cfg_trainer['use_tqdm']:
                tqdm_callback = Tqdm(epoch, len(self.datamanager.get_dataloader('val')), phase='val')
            for batch_idx, (data, labels, attribute_idx) in enumerate(self.datamanager.get_dataloader('val')):
                if not self.cfg_trainer['use_tqdm']:
                    start_time = time.time()
                # push data to device
                data, labels, attribute_idx = data.to(self.device), labels.to(self.device), attribute_idx.to(self.device)
                
                # forward batch
                out = self.model(data)

                # calculate loss and accuracy
                loss = self.criterion(out, labels, attribute_idx)

                # calculate instance-based accuracy
                preds = torch.sigmoid(out)
                
                mean_accuracy, accuracy, f1_score = compute_accuracy_cuda(labels, preds)

                # update loss and accuracy in MetricTracker
                self.valid_metrics.update('loss', loss.item())
                self.valid_metrics.update('mA', mean_accuracy)
                self.valid_metrics.update('accuracy', accuracy)
                self.valid_metrics.update('f1_score', f1_score)

                # update process
                if self.cfg_trainer['use_tqdm']:
                    tqdm_callback.on_batch_end({
                        'loss': loss.item(),
                        'mA': mean_accuracy,
                        'accuracy': accuracy,
                        'f1-score': f1_score})
                else:
                    end_time = time.time()
                    if (batch_idx+1) % self.log_step[1] == 0 or (batch_idx+1) == len(self.datamanager.get_dataloader('val'))-1:
                        self.logger.info('Valid Epoch: {} {}/{} {:.1f}batch/s Loss: {:.4f} mA: {:.4f} Acc: {:.4f} F1-score: {:.4f}'.format(
                            epoch,
                            batch_idx+1,
                            len(self.datamanager.get_dataloader('val')),
                            1/(end_time-start_time),
                            loss.item(),
                            mean_accuracy,
                            accuracy,
                            f1_score))
        if self.cfg_trainer['use_tqdm']:
            tqdm_callback.on_epoch_end()
        return self.valid_metrics.result()
