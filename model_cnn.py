import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from wettbewerb import load_references, get_3montages, get_6montages
import mne
from scipy import signal as sig
from imblearn.under_sampling import RandomUnderSampler

class CNN(nn.Module):
    def __init__(self, num_classes, seq_length):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Conv1d(in_channels=3, out_channels=6, kernel_size=5),
            nn.BatchNorm1d(num_features=6),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(6, 16, 5),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2, 2),
        )

        # Anpassung für die Berechnung der Größe des linearen Layers
        linear_input_size = self._get_conv_output(seq_length)

        self.fc = nn.Sequential(
            nn.Linear(linear_input_size, 120),
            nn.BatchNorm1d(120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.BatchNorm1d(84),
            nn.ReLU(),
            nn.Linear(84, num_classes),
        )

    def _get_conv_output(self, shape):
        with torch.no_grad():
            input = torch.zeros(1, 3, shape)
            output = self.classifier(input)
            return output.numel()

    def forward(self, x):
        x = self.classifier(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return x
    
if __name__ == '__main__':
    print("hey")