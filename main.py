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

import argparse
from dataset import GetDataset
from torch.utils.data import DataLoader, random_split
import torch
from torchvision import transforms
from train import Trainer
from run import RunExperiment
from network import Model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train', help='choose to train or test the model')

    print('>>>> Loading Dataset')
    dataset = GetDataset(root_dir='/home/arafanelli/PycharmProjects/emoDetection/RAF-DB/DATASET', batch_size=64, num_workers=4)
    train_loader, val_loader = dataset.get_data_loaders()
    print('>>>> Dataset Loaded ')


    experiment = RunExperiment('emotion', train_loader, val_loader)

    if args.mode == 'train':
        experiment.run()

    elif args.mode == 'test':
        experiment.load_best_model()

    else:
        print("Invalid argument. Please enter 'train' or 'test'.")




