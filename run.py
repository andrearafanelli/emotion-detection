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
from train import Trainer, Tester
from network import Model
import copy
import os


class RunExperiment:

    def __init__(self, name, train, test, num_epochs=200, learning_rate=1e-1,
                 momentum=9e-1):
        self.train_loader = train
        self.test_loader = test
        self.best_model = None
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.name = name
        self.best_loss = 1e10
        self.best_acc = 0
        self.best_epoch = 0

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = Model().to(self.device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=self.momentum)

    def run(self):
        os.makedirs(f"{os.getcwd()}/models/", exist_ok=True)
        torch.cuda.empty_cache()
        for epoch in range(self.num_epochs):
            print('*' * 40)
            print('Epoch {}/{}'.format(epoch, self.num_epochs - 1))
            print('*' * 40)
            trainer = Trainer(self.model, self.optimizer, self.device)
            train_metrics = trainer.training(self.train_loader, 'Train')
            test_metrics = trainer.testing(self.test_loader, 'Test')

            if (test_metrics['Accuracy'] >= self.best_acc) and (test_metrics['Loss'] <= self.best_loss):
                self.best_acc = test_metrics['Accuracy']
                self.best_epoch = epoch
                self.best_loss = test_metrics['Loss']
                self.best_model = copy.deepcopy(self.model.state_dict())
                print(f'Best Accuracy: {self.best_acc:.4f} Epoch: {epoch + 1}')
                print(">>>>> Saving model..")
                torch.save(self.best_model, f"{os.getcwd()}/models/{self.name}.pt")

    def load_best_model(self):
        self.model.load_state_dict(torch.load(f"{os.getcwd()}/models/{self.expName}.pt"))
        trainer = Tester(self.model, self.optimizer, self.device)
        train_epoch = trainer.testing(self.val_loader, 'Validation')
