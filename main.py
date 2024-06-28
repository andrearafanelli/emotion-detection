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
from configuration import Config
from utils.dataset import GetDataset
from run import RunExperiment


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train', help='choose to train or test the model')
    args = parser.parse_args()
    config_param = Config

    print('>>>> Loading Dataset')
    dataset = GetDataset(config_param)
    train_loader, val_loader, test_loader = dataset.get_data_loaders()
    print('>>>> Dataset Loaded ')


    experiment = RunExperiment('26_02_17_00', train_loader, val_loader, test_loader, config_param)

    if args.mode == 'train':
        print('>>>> Starting Experiments ')
        experiment.run()

    elif args.mode == 'test':
        print('>>>> Starting Evaluation ')
        experiment.load_best_model()

    else:
        print("Invalid argument. Please enter 'train' or 'test'.")
