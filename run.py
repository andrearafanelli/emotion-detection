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
from train import Trainer
from network import model
import copy
import os
import time
import wandb


class RunExperiment:

    def __init__(self, name, train, test, configuration):
        self.train_loader = train
        self.test_loader = test
        self.num_epochs = configuration['num_epochs']
        self.learning_rate = configuration['learning_rate']
        self.momentum = configuration['momentum']
        self.weight_decay = configuration['weight_decay']
        self.name = name
        self.best_model = None
        self.best_loss = 1e10
        self.best_acc = 0
        self.best_epoch = 0

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = torch.nn.DataParallel(model()).to(self.device)
        print('>>>> Model initialized!')
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate,
                                         momentum=self.momentum, weight_decay=self.weight_decay)

    def run(self):
        wandb.login()
        wandb.init(project='emotion-detection')

        metrics_list = []
        os.makedirs(f"{os.getcwd()}/models/", exist_ok=True)
        torch.cuda.empty_cache()
        for epoch in range(self.num_epochs):
            print('*' * 40)
            print('Epoch {}/{}'.format(epoch, self.num_epochs - 1))
            print('*' * 40)
            trainer = Trainer(self.model, self.optimizer, self.device)
            train_metrics = trainer.training(self.train_loader, 'Train')
            test_metrics = trainer.testing(self.test_loader, 'Test')

            epoch_metrics = {'Train Loss': train_metrics['Loss'], 'Train Accuracy': train_metrics['Accuracy'],
                             'Train F1': train_metrics['F1'], 'Train Precision': train_metrics['Precision'],
                             'Train Recall': train_metrics['Recall'],
                             'Test Loss': test_metrics['Loss'], 'Test Accuracy': test_metrics['Accuracy'],
                             'Test F1': test_metrics['F1'], 'Test Precision': test_metrics['Precision'],
                             'Test Recall': test_metrics['Recall']
                             }
            wandb.log(epoch_metrics)
            metrics_list.append(epoch_metrics)

            if (test_metrics['Accuracy'] >= self.best_acc) and (test_metrics['Loss'] <= self.best_loss):
                self.best_acc = test_metrics['Accuracy']
                self.best_epoch = epoch
                self.best_loss = test_metrics['Loss']
                self.best_model = copy.deepcopy(self.model.state_dict())
                print(f'Best Accuracy: {self.best_acc:.4f} Epoch: {epoch + 1}')
                print(">>>> Saving model..")
                torch.save(self.best_model, f"{os.getcwd()}/models/{self.name}.pt")

        metrics_df = pd.DataFrame(metrics_list)
        metrics_df.to_csv('experiment_metrics.csv', index=False)

    def load_best_model(self):
        self.model.load_state_dict(torch.load(f"{os.getcwd()}/models/{self.expName}.pt"))
        trainer = Tester(self.model, self.optimizer, self.device)
        train_epoch = trainer.testing(self.val_loader, 'Validation')
