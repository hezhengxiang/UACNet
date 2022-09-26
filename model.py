import torch
import torch.nn as nn
import torch.nn.functional as F

"""
    Author: Zhengxiang He.
    Date: 25 June 2022
    Description: 
"""


class UpdateGate(nn.Module):
    """
        Description: A Gate for determine the output will be upgrade or not.
        in_features: the number of input's feature.
        out_features: the number of output's feature.
    """
    def __init__(self, in_features, out_features):
        super(UpdateGate, self).__init__()
        self.fc = nn.Linear(in_features=in_features, out_features=out_features)

    def forward(self, x):
        z = x.squeeze(0).T
        z = torch.sigmoid(self.fc(z))
        return z.T.unsqueeze(0)


class ResetGate(nn.Module):
    def __init__(self, in_features, out_features):
        super(ResetGate, self).__init__()
        self.fc = nn.Linear(in_features=in_features, out_features=out_features)
        self.bias = nn.Parameter(torch.zeros((1, out_features, 1), requires_grad=True))

    def forward(self, x):
        r = (x.squeeze(0)).T
        r = torch.sigmoid(self.fc(r))
        return torch.relu(r.T.unsqueeze(0) * x + self.bias)


class BasicBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        # self.bn1 = nn.InstanceNorm1d(out_channels)
        self.bn1 = nn.InstanceNorm1d(out_channels)
        self.conv2 = nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        # self.bn2 = nn.InstanceNorm1d(out_channels)
        self.bn2 = nn.InstanceNorm1d(out_channels)
        self.ug = UpdateGate(in_features=out_channels, out_features=out_channels)
        self.rg = ResetGate(in_features=in_channels, out_features=out_channels)

    def forward(self, x):
        r = self.rg(x)
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        z = self.ug(x)
        y = z * x + (1 - z) * r
        # y2 = F.adaptive_avg_pool1d(y, 15)
        return y


class UACNet(nn.Module):

    def __init__(self):
        super(UACNet, self).__init__()
        self.conv_block1 = nn.Sequential(*[
                        nn.Conv1d(1, 64, 7, 1, 3),
                        nn.InstanceNorm1d(64)
                        # nn.LayerNorm(64),
                        ])
        self.basic_block1 = nn.Sequential(*[
            BasicBlock(64, 64),
            BasicBlock(64, 64),
            # BasicBlock(64, 64)
        ])
        self.conv_block2 = nn.Sequential(*[
            nn.Conv1d(64, 128, 7, 1, 3),
            nn.InstanceNorm1d(128)
            # nn.LayerNorm(128),
        ])
        self.basic_block2 = nn.Sequential(*[
            BasicBlock(128, 128),
            BasicBlock(128, 128),
            # BasicBlock(128, 128)
        ])
        self.conv_block3 = nn.Sequential(*[
            nn.Conv1d(128, 256, 7, 1, 3),
            nn.InstanceNorm1d(256)
            # nn.LayerNorm(256),
        ])
        self.basic_block3 = nn.Sequential(*[
            BasicBlock(256, 256),
            BasicBlock(256, 256),
            # BasicBlock(256, 256)
        ])
        self.conv_block4 = nn.Sequential(*[
            nn.Conv1d(256, 512, 7, 1, 3),
            nn.InstanceNorm1d(512)
            # nn.LayerNorm(512),
        ])
        self.basic_block4 = nn.Sequential(*[
            BasicBlock(512, 512),
            BasicBlock(512, 512),
            # BasicBlock(512, 512)
        ])
        self.conv_block5 = nn.Sequential(*[
            nn.Conv1d(512, 1024, 7, 1, 3),
            nn.InstanceNorm1d(1024)
            # nn.LayerNorm(1024),
        ])
        self.basic_block5 = nn.Sequential(*[
            BasicBlock(1024, 1024),
            BasicBlock(1024, 1024),
            # BasicBlock(1024, 1024),
        ])
        self.basic_block6 = nn.Sequential(*[
            BasicBlock(1024, 1024),
            BasicBlock(1024, 1024),
            # BasicBlock(1024, 1024),
        ])
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 5)
        self.mp1 = nn.MaxPool1d(7, stride=2, padding=3)
        self.mp2 = nn.MaxPool1d(7, stride=2, padding=3)
        self.mp3 = nn.MaxPool1d(7, stride=2, padding=3)
        # self.dropout1 = nn.Dropout(0.9)
        # self.mp1 = nn.AvgPool1d(3, stride=2)
        # self.mp2 = nn.AvgPool1d(3, stride=2)
        # self.mp3 = nn.AvgPool1d(3, stride=2)

    def forward(self, x1):
        x = torch.relu(self.conv_block1(x1))
        x = self.basic_block1(x)
        x = self.mp1(x)
        x = torch.relu(self.conv_block2(x))
        x = self.basic_block2(x)
        x = self.mp2(x)
        x = torch.relu(self.conv_block3(x))
        x = self.basic_block3(x)
        x = self.mp3(x)
        x = torch.relu((self.conv_block4(x)))
        x = self.basic_block4(x)
        x = F.adaptive_avg_pool1d(x, 64)
        x = torch.relu((self.conv_block5(x)))
        x = (self.basic_block5(x))
        # x = self.mp4(x)
        # x = (self.basic_block6(x))
        x = F.adaptive_avg_pool1d(x, 1)
        x = torch.flatten(x)
        x = torch.relu(F.dropout(self.fc1(x), p=0.7))
        y = (self.fc2(x))

        return y
