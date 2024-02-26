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
from metrics import Metrics, calculate_accuracy, Score
import pickle


class Trainer:

    def __init__(self, model, optimizer, device):

        self.accuracy = None
        self.loss = None
        self.metrics_calculator = None
        self.model = model
        self.optimizer = optimizer
        self.device = device

    def training(self, dataloaders, set_name):
        self.accuracy, self.loss, self.metrics_calculator = Metrics(), Metrics(), Score()
        criterion = torch.nn.CrossEntropyLoss()

        self.model.train()

        for batch_idx, (inputs, targets) in enumerate(tqdm(dataloaders)):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            outputs = self.model(inputs)
            loss = criterion(outputs, targets)

            accuracy = calculate_accuracy(outputs, targets)

            self.accuracy.average(accuracy, inputs)
            self.loss.average(loss, inputs)
            self.metrics_calculator.collect(targets, outputs)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return self.metrics_calculator.calculate(self.accuracy.avg, self.loss.avg, set_name)

    def testing(self, dataloaders, set_name):
        self.accuracy, self.loss, self.metrics_calculator = Metrics(), Metrics(), Score()
        criterion = torch.nn.CrossEntropyLoss()
        results = []

        self.model.eval()

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(tqdm(dataloaders)):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                outputs = self.model(inputs)

                results.append(
                    {'ID': inputs.imgs[0][0], 'Ground Truth': targets.item(), 'Prediction': outputs.item()})

                loss = criterion(outputs, targets)
                accuracy = calculate_accuracy(outputs, targets)

                self.accuracy.average(accuracy, inputs)
                self.loss.average(loss, inputs)
                self.metrics_calculator.collect(targets, outputs)

        with open('predictions.pkl', 'wb') as f:
            pickle.dump(results, f)

        return self.metrics_calculator.calculate(self.accuracy.avg, self.loss.avg, set_name)



