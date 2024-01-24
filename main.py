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

    train = GetDataset3(root_dir='images/train/', train=True)
    test = GetDataset3(root_dir='images/validation/', train=False)

    val_size = int(0.3 * len(train))

    train, val = random_split(train, [len(train) - val_size, val_size])

    train_loader = DataLoader(train, batch_size=16, shuffle=True)
    val_loader = DataLoader(val, batch_size=16, shuffle=False)
    test_loader = DataLoader(test, batch_size=16, shuffle=False)

    experiment = RunExperiment('emotion', train_loader, val_loader)

    if args.mode == 'train':
        experiment.run()

    elif args.mode == 'test':
        experiment.load_best_model()

    else:
        print("Invalid argument. Please enter 'train' or 'test'.")




