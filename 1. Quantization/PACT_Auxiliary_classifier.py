import torch
import torch.nn as nn
import torch.nn.functional as F

class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.aux_classifier = nn.Linear(out_channels, num_classes)  # Auxiliary classifier

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # Calculate auxiliary classifier loss
        aux_out = F.avg_pool2d(out, out.size()[2:])
        aux_out = aux_out.view(aux_out.size(0), -1)
        aux_out = self.aux_classifier(aux_out)     #layer마다 중간에 classifier를 진행 
                                                   #그렇다면 label 필요
        out += identity
        out = self.relu(out)

        return out, aux_out

class ResNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        self.block1 = ResNetBlock(64, 64)
        self.block2 = ResNetBlock(64, 64)
        self.block3 = ResNetBlock(64, 64)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x, aux_out1 = self.block1(x)   #block별로 auxiliary classifier result 출력
        x, aux_out2 = self.block2(x)
        x, aux_out3 = self.block3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x, aux_out1, aux_out2, aux_out3

# Usage example
num_classes = 10
model = ResNet(num_classes=num_classes)
criterion = nn.CrossEntropyLoss()

# Assuming you have the input data 'inputs' and corresponding labels 'targets'
outputs, aux1, aux2, aux3 = model(inputs)       #각 label들 input으로 입력
loss1 = criterion(aux1, targets)                #각각의 loss 값 출력
loss2 = criterion(aux2, targets)
loss3 = criterion(aux3, targets)
main_loss = criterion(outputs, targets)

# Total loss is a combination of the main loss and the auxiliary losses
total_loss = main_loss + loss1 + loss2 + loss3

# Perform backpropagation and update the model parameters
total_loss.backward()
optimizer.step()
