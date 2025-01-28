import torch
import torch.nn as nn
import torch.nn.functional as F

class FootballMatchPredictor(nn.Module):
    def __init__(self, input_agg_stats_size, hidden_size=64):
        super(FootballMatchPredictor, self).__init__()
        self.agg_stats_fc = nn.Sequential(
            nn.Linear(input_agg_stats_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU()
        )
        combined_input_size = hidden_size // 2
        self.fc_combined = nn.Sequential(
            nn.Linear(combined_input_size, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, 2)
        )

    def forward(self, agg_stats):
        agg_stats_out = self.agg_stats_fc(agg_stats)
        output = self.fc_combined(agg_stats_out)
        return F.softmax(output, dim=1)

class FootballMatchPredictorOutcome(nn.Module):
    def __init__(self, input_agg_stats_size, hidden_size=128):
        super(FootballMatchPredictorOutcome, self).__init__()
        self.agg_stats_fc = nn.Sequential(
            nn.Linear(input_agg_stats_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        combined_input_size = hidden_size
        self.fc_combined = nn.Sequential(
            nn.Linear(combined_input_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 3)
        )

    def forward(self, agg_stats):
        agg_stats_out = self.agg_stats_fc(agg_stats)
        output = self.fc_combined(agg_stats_out)
        return F.softmax(output, dim=1)
