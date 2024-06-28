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

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.utils.multiclass import unique_labels
import torch
import matplotlib.pyplot as plt
import seaborn as sns


def calculate_accuracy(predictions, targets):
    with torch.no_grad():
        batch_size = targets.size(0)
        predicted_labels = torch.argmax(predictions, dim=1)
        correct_predictions = predicted_labels.eq(targets)
        accuracy = correct_predictions.float().sum().mul_(1.0 / batch_size)

    return accuracy


class Metrics:
    def __init__(self):
        self.avg, self.sum, self.count = 0, 0, 0

    def average(self, metric, input):
        sample = input.size(0)
        self.sum += metric.item() * sample
        self.count += sample
        if self.count != 0:
            self.avg = self.sum / self.count


class Score:
    def __init__(self):
        self.y_true = []
        self.y_pred = []

    def collect(self, y_true_batch, y_pred_batch):
        self.y_true.append(y_true_batch)
        self.y_pred.append(torch.argmax(y_pred_batch, dim=1))

    def calculate(self, accuracy, loss, phase):
        metrics = {}
        y_pred = torch.cat(self.y_pred).cpu().numpy()
        y_true = torch.cat(self.y_true).cpu().numpy()
        labels = unique_labels(y_true, y_pred)

        metrics['Accuracy'] = accuracy
        metrics['Loss'] = loss
        metrics['Recall'] = recall_score(y_true, y_pred, average='macro', zero_division=1.0)
        metrics['Precision'] = precision_score(y_true, y_pred, average='macro', zero_division=1.0)
        f1_scores = f1_score(y_true, y_pred, average=None, labels=labels)
        metrics['F1'] = f1_scores.sum() / labels.shape[0]
        metrics['ConfusionMatrix'] = confusion_matrix(y_true, y_pred)

        self.print_metrics(metrics, phase)
        return metrics

    @staticmethod
    def print_metrics(metrics, phase):
        outputs = []
        try:
            for k in metrics.keys():
                if k != 'Confusion Matrix':
                    outputs.append("{}: {:4f}".format(k, metrics[k]))

            print("{}: {}".format(phase, ", ".join(outputs)))
        except:
            pass


def plot(metrics):
    class_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    plt.figure(figsize=(12, 5))
    plt.rcParams["font.size"] = 12
    plt.rcParams["font.weight"] = "bold"
    sns.heatmap(metrics['ConfusionMatrix'], annot=True, fmt='d', cmap='Blues', xticklabels=class_names,
                yticklabels=class_names)
    plt.xlabel('\nPredicted', size=15, color='orangered')
    plt.ylabel('Ground Truth\n', size=15, color='dodgerblue')
    plt.title('Confusion Matrix', size=20, color='teal', fontweight="bold")

    plt.text(0.3, -0.5, f"Test Accuracy: {100 * metrics['Accuracy']:.2f} %", fontsize=12, color='crimson',
             bbox=dict(facecolor='white', edgecolor='crimson', boxstyle='round,pad=0.4'))

    plt.show()
