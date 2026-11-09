"""CNN-1D baseline, trained from scratch; outputs unnormalized logits."""
from torch import nn


class CNN1D(nn.Module):
    def __init__(self, input_channels=5, num_classes=3):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(input_channels, 32, 7, padding=3), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 5, padding=2), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 3, padding=1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, num_classes))

    def forward(self, x):
        return self.network(x)
