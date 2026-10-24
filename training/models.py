from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn

from env.encoding import SHAPE


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(SHAPE[0], 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 11 * 11, 128),
            nn.ReLU(),
        )

    def forward(self, obs):
        return self.layers(obs.float() / 255.0)


class Extractor(BaseFeaturesExtractor):
    def __init__(self, observation_space):
        super().__init__(observation_space, 128)
        self.cnn = CNN()

    def forward(self, obs):
        return self.cnn(obs)


class PPOExport(nn.Module):
    def __init__(self, policy):
        super().__init__()
        self.cnn = policy.features_extractor.cnn
        self.actor = policy.action_net
        self.value = policy.value_net

    def forward(self, obs, hidden):
        features = self.cnn(obs)
        return self.actor(features), self.value(features).squeeze(-1), hidden


def masked(scores, mask):
    return scores.masked_fill(~mask.bool(), -1e9)
