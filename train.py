'''
Copyright 2024 Andrea Rafanelli.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and limitations under the License
'''

__author__ = 'Andrea Rafanelli'

import torch
import os
import numpy as np
import torch.nn.functional as F
import sklearn
from collections import defaultdict
from tqdm.auto import tqdm
from metrics import MetricsCalculator


class Trainer:

    def __init__(self, model, optimizer, device):

        self.metrics_calculator = None
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.metrics = defaultdict(float)
        self.loss_function = LossCalculator()

    def training(self, dataloaders, set_name):
        self.metrics.clear()
        epoch_samples = 0
        criterion = torch.nn.CrossEntropyLoss(reduction='mean')
        self.model.train()

        for batch_idx, (inputs, targets) in enumerate(tqdm(dataloaders)):
            inputs = inputs.to(self.device)
            targets = torch.LongTensor(targets).to(self.device)
            self.optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                self.metrics_calculator = MetricsCalculator(targets, outputs, loss)
                self.metrics = self.metrics_calculator.calculate_metrics(self.metrics)
                loss.backward()
                self.optimizer.step()

            epoch_samples += inputs.size(0)

        self.metrics_calculator._update_metrics(self.metrics, epoch_samples, set_name)

        return self.metrics

    def testing(self, dataloaders, set_name):
        self.metrics.clear()
        criterion = torch.nn.CrossEntropyLoss(reduction='mean')
        epoch_samples = 0
        self.model.eval()
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(tqdm(dataloaders)):
                inputs = inputs.to(self.device)
                targets = torch.LongTensor(targets).to(self.device)
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                self.metrics_calculator = MetricsCalculator(targets, outputs, loss)
                self.metrics = self.metrics_calculator.calculate_metrics(self.metrics)
                epoch_samples += inputs.size(0)

        self.metrics_calculator._update_metrics(self.metrics, epoch_samples, set_name)

        return self.metrics


class Tester:
    """Initialize the tester with the model, optimizer, device,
     batch, valid_path, metrics, metrics_calculator, and loss_function"""

    def __init__(self, model, optimizer, device):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.batch = 32
        self.metrics_calculator = MetricsCalculator()
        self.loss_function = LossCalculator()
        self.metrics = defaultdict(float)

    def testing(self, dataloaders, set_name):
        """Test the model on the provided data loader,
        save the output masks, and update the metrics dictionary"""
        self.metrics.clear()
        self.model.eval()
        epoch_samples = 0

        for index, inputs, labels in tqdm(dataloaders):
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            with torch.no_grad():
                outputs = self.model(inputs)
                preds = torch.sigmoid(outputs)
                preds = preds.data.cpu().numpy()
                loss = self.loss_function(outputs, labels, self.metrics)
                epoch_samples += inputs.size(0)

        self.metrics_calculator._update_metrics(self.metrics, epoch_samples, set_name)

        return self.metrics


class LossCalculator:

    def __init__(self, bce_weight=0.35):
        self.bce_weight = bce_weight

    def __call__(self, y_pred, y_true, metrics):
        loss = torch.nn.CrossEntropyLoss(reduction='mean')
        loss = loss(y_pred, y_true)

        metrics = defaultdict(float)
        calculator = MetricsCalculator(y_true, y_pred)

        metrics['Loss'] += loss * y_true.size(0)
        metrics['Accuracy'] += calculator.calculate_accuracy() * y_true.size(0)
        metrics['F1'] += calculator.calculate_f1_score() * y_true.size(0)
        metrics['Precision'] += calculator.calculate_precision() * y_true.size(0)
        metrics['Recall'] += calculator.calculate_recall() * y_true.size(0)

        pred_prob = F.softmax(pred, dim=1)

        target_np = target.data.cpu().numpy()
        pred_np = pred.data.cpu().numpy()

        MIoU = np.mean(
            jaccard_score(np.argmax(target_np, axis=1).flatten(), np.argmax(pred_np, axis=1).flatten(), average=None))
        accuracy = accuracy_score(np.argmax(target_np, axis=1).flatten(), np.argmax(pred_np, axis=1).flatten())
        self._update_metrics(metrics, target.size(0), bce, dice, loss, accuracy, MIoU)
        return loss

    def _update_metrics(self, metrics, batch_size, bce, dice, loss, accuracy, MIoU):
        metrics['Bce'] += bce.data.cpu().numpy() * batch_size
        metrics['Dice'] += dice.data.cpu().numpy() * batch_size
        metrics['Loss'] += loss.data.cpu().numpy() * batch_size
        metrics['Accuracy'] += accuracy * batch_size
        metrics['MIoU'] += MIoU * batch_size
