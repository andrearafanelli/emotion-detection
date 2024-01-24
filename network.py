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
import torch.nn as nn
from torchvision import models


class Model(nn.Module):
    def __init__(self, num_classes=7):
        super(Model, self).__init__()
        model = models.mobilenet_v2(pretrained=True)
        model.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.features = model.features
        self.global_avg_pooling = nn.AdaptiveAvgPool2d(1)
        self.dense1 = nn.Linear(1280, 512)
        self.relu1 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(0.25)
        self.dense2 = nn.Linear(512, 256)
        self.relu2 = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout(0.25)
        self.dense3 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.global_avg_pooling(x)
        x = x.view(x.size(0), -1)
        x = self.dense1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.dense2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.dense3(x)
        return x
