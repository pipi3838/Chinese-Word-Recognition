import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as  F

class Resnet(nn.Module):
    def __init__(self, num_class):
        super(Resnet, self).__init__()
        self.resnet = models.resnet50(pretrained=True)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_class)
    
    def forward(self, x):
        return self.resnet(x)

def Conv3x3BNReLU(in_channels,out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(0.2, inplace=True)
    )

class Densenet(nn.Module):
    def __init__(self, num_class):
        super(Densenet, self).__init__()
        self.net = models.densenet121(pretrained=True)
        # torch.hub.load('pytorch/vision:v0.9.0', 'densenet121', pretrained=True)
        self.net.classifier = nn.Linear(self.net.classifier.in_features, num_class)
    
    def forward(self, x):
        return self.net(x)

# class PretrainDensenet(nn.Module):
#     def __init__(self, num_class):
#         super(PretrainDensenet, self).__init__()
#         self.net = models.densenet121(pretrained=False)
#         self.net.classifier = nn.Linear(self.net.classifier.in_features, num_class)
    
#     def forward(self, x):
#         return self.net(x)

def Conv3x3BNReLU(in_channels,out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(0.2, inplace=True)
    )
class Net(nn.Module):
    def __init__(self, num_class):
        super(Net, self).__init__()
        # self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1)
        # self.pool = nn.MaxPool2d(2, 2)
        # self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        # self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        # self.conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.conv1 = Conv3x3BNReLU(1, 64)
        self.conv2 = Conv3x3BNReLU(64, 128)
        self.conv3 = Conv3x3BNReLU(128, 256)
        self.conv4 = Conv3x3BNReLU(256, 512)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(4608, 8192)
        self.fc2 = nn.Linear(8192, 4096)
        self.fc3 = nn.Linear(4096, num_class)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        # x = self.pool(F.relu(self.conv1(x)))
        # x = self.pool(F.relu(self.conv2(x)))
        # x = self.pool(F.relu(self.conv3(x)))
        # x = self.pool(F.relu(self.conv4(x)))
        x = self.pool(self.conv1(x))
        x = self.pool(self.conv2(x))
        x = self.pool(self.conv3(x))
        x = self.pool(self.conv4(x))
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return x

    @staticmethod
    def num_flat_features(x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

# class Net(nn.Module):
#     def __init__(self, num_class, block_nums=[1,1,1,1]):
#         super(Net, self).__init__()
#         self.stage1 = self._make_layers(in_channels=1, out_channels=64, block_num=block_nums[0])
#         self.stage2 = self._make_layers(in_channels=64, out_channels=128, block_num=block_nums[1])
#         self.stage3 = self._make_layers(in_channels=128, out_channels=256, block_num=block_nums[2])
#         # self.stage4 = self._make_layers(in_channels=256, out_channels=512, block_num=block_nums[3])
#         self.fc1 = nn.Linear(9216, 4096)
#         self.fc2 = nn.Linear(4096, 4096)
#         self.fc3 = nn.Linear(4096, num_class)
#         self.dropout = nn.Dropout(0.2)
#         # self._init_params()

#     def forward(self, x):
#         x = self.stage1(x)
#         x = self.stage2(x)
#         x = self.stage3(x)
#         # x = self.stage4(x)
#         x = x.view(x.size(0), -1)
#         x = self.dropout(self.fc1(x))
#         x = self.dropout(self.fc2(x))
#         x = self.dropout(self.fc3(x))

#         return x

#     def _init_params(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)

#     def _make_layers(self, in_channels, out_channels, block_num):
#         layers = []
#         layers.append(Conv3x3BNReLU(in_channels,out_channels))
#         for i in range(1, block_num):
#             layers.append(Conv3x3BNReLU(out_channels, out_channels))
#         layers.append(nn.MaxPool2d(kernel_size=2,stride=2, ceil_mode=False))
        
#         return nn.Sequential(*layers)