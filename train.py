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

    def __init__(self, model, optimizer, scheduler, device):

        self.metrics_calculator = None
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device

    def training(self, dataloaders, set_name):
        self.metrics = defaultdict(float)
        epoch_samples = 0
        criterion = torch.nn.CrossEntropyLoss()
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
                self.scheduler.step()

            epoch_samples += inputs.size(0)

        self.metrics_calculator._update_metrics(self.metrics, epoch_samples, set_name)

        return self.metrics

    def testing(self, dataloaders, set_name):
        self.metrics = defaultdict(float)
        criterion = torch.nn.CrossEntropyLoss()
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


