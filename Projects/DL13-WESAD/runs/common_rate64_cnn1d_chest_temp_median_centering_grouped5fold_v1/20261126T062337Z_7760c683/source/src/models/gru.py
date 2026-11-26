"""E2: a single-layer, unidirectional GRU over the full resampled window."""
from torch import nn


class GRUClassifier(nn.Module):
    def __init__(self, input_channels=5, num_classes=3, hidden_size=64, dropout=0.3):
        super().__init__()
        if isinstance(hidden_size, bool) or not isinstance(hidden_size, int) or hidden_size <= 0:
            raise ValueError('GRU hidden_size must be a positive integer.')
        if not 0 <= dropout < 1:
            raise ValueError('GRU dropout must be in [0, 1).')
        self.gru = nn.GRU(input_channels, hidden_size, batch_first=True)
        # nn.GRU dropout applies only between recurrent layers. This one-layer
        # model uses explicit classifier dropout, like the CNN classifier.
        self.classifier = nn.Sequential(nn.Linear(hidden_size, 64), nn.ReLU(),
                                        nn.Dropout(dropout), nn.Linear(64, num_classes))

    def forward(self, x):
        # Dataset layout [batch, channel, time] -> GRU [batch, time, feature].
        _, hidden = self.gru(x.transpose(1, 2).contiguous())
        return self.classifier(hidden[-1])
