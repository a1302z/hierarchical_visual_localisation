import torch
import torch.nn as nn

class Classifier(nn.Module):
    def __init__(self, pretrained_model, input_channels=1):
        super(Classifier, self).__init__()
        self.pt = pretrained_model
        if input_channels != 3:
            self.pt.conv1 = torch.nn.Conv2d(input_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.pt.avgpool = nn.AdaptiveAvgPool2d(1)
        
    def forward(self, x):
        return self.pt(x)
    
    
    
if __name__ == '__main__':
    tm = Classifier(lambda x: x)
    tm(0)
    print('Test successful')