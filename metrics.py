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

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch


class MetricsCalculator:
    def __init__(self, y_true, y_pred, loss):
        self.y_true = y_true.data.cpu().numpy()
        self.y_pred = torch.argmax(y_pred, axis=1).data.cpu().numpy()
        self.loss = loss

    def calculate_accuracy(self):
        return accuracy_score(self.y_true, self.y_pred)

    def calculate_precision(self, average='macro'):
        return precision_score(self.y_true, self.y_pred, average=average, zero_division=1.0)

    def calculate_recall(self, average='macro'):
        return recall_score(self.y_true, self.y_pred, average=average, zero_division=1.0)

    def calculate_f1_score(self, average='macro'):
        return f1_score(self.y_true, self.y_pred, average=average, zero_division=1.0)

    def calculate_metrics(self, metrics):
        batch_size = self.y_true.shape[0]
        metrics['Loss'] += self.loss * batch_size
        metrics['Accuracy'] += self.calculate_accuracy() * batch_size
        metrics['F1'] += self.calculate_f1_score() * batch_size
        metrics['Precision'] += self.calculate_precision() * batch_size
        metrics['Recall'] += self.calculate_recall() * batch_size
        return metrics

    def _update_metrics(self, metrics, epoch_samples, set_name):
        for metric in ['Loss', 'Accuracy', 'F1', 'Precision', 'Recall']:
            metrics[metric] /= epoch_samples

        self.print_metrics(metrics, set_name)

    @staticmethod
    def print_metrics(metrics, phase):
        outputs = []
        for k in metrics.keys():
            outputs.append("{}: {:4f}".format(k, metrics[k]))
        print("{}: {}".format(phase, ", ".join(outputs)))


# Example of how to use the EmotionMetrics class
if __name__ == "__main__":
    # Example data (replace with your actual data)
    y_true = [1, 0, 1, 0, 1, 1, 0, 0, 1, 0]
    y_pred = [1, 1, 1, 0, 0, 0, 0, 0, 1, 0]
    y_pred_prob = [0.8, 0.3, 0.6, 0.2, 0.7, 0.9, 0.4, 0.1, 0.75, 0.25]

    # Instantiate the EmotionMetrics class
    emotion_metrics = MetricsCalculator(y_true, y_pred_prob, y_pred)

    # Calculate and print various metrics
    print(f'Accuracy: {emotion_metrics.calculate_accuracy()}')
    print(f'Precision: {emotion_metrics.calculate_precision()}')
    print(f'Recall: {emotion_metrics.calculate_recall()}')
    print(f'F1 Score: {emotion_metrics.calculate_f1_score()}')
    print(f'Confusion Matrix:\n{emotion_metrics.calculate_confusion_matrix()}')
    print(f'AUC: {emotion_metrics.calculate_auc()}')
    print(f'Mean Absolute Error: {emotion_metrics.calculate_mean_absolute_error()}')
    print(f'Mean Squared Error: {emotion_metrics.calculate_mean_squared_error()}')
    print(f'Intersection over Union (IoU): {emotion_metrics.calculate_iou()}')

    # Plot ROC Curve
    emotion_metrics.plot_roc_curve()
