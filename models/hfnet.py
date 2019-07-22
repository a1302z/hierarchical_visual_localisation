import torch
import torch.nn as nn
import torchvision.models as models
import models.netvlad_vd16_pitts30k_conv5_3_max_dag as netvlad
import models.demo_superpoint as superpoint

from torchsummary import summary

class hfnet(nn.Module):
    def __init__(self):
        super(hfnet, self).__init__()
        #backbone = models.resnet34(pretrained=True)
        #self.extractor = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool, backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4)
        self.extractor = models.mobilenet_v2(pretrained=True)
        #self.netvlad = netvlad.Vd16_pitts30k_conv5_3_max_dag()
        self.superpoint = superpoint.SuperPointNet(c0 = 64)
        print('Init done')
        
    def forward(self, x):
        x = self.extractor.features(x)
        print(x.size())
        vlad = x.view(x.size(0), -1)
        print(vlad.size())
        return x, vlad

"""
class Classifier(nn.Module):
    def __init__(self, pretrained_model, input_channels=1):
        super(Classifier, self).__init__()
        self.pt = pretrained_model
        if input_channels != 3:
            self.pt.conv1 = torch.nn.Conv2d(input_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.pt.avgpool = nn.AdaptiveAvgPool2d(1)
        
    def forward(self, x):
        return self.pt(x)
    
"""   
    
if __name__ == '__main__':
    hf = hfnet()
    #print(hf)
    if torch.cuda.is_available():
        hf = hf.cuda()
    hf.eval()
    print(summary(hf, (3,1600,1093)))
    print('Test successful')