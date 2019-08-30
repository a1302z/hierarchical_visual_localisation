import torch
import torch.nn as nn
from math import sqrt, log
from torchsummary import summary


def weight_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)

def init_model_weights(model):
    print('Initializing model weights')
    for m in model.modules():
        weight_init(m)
        if isinstance(m, nn.ModuleList):
            for md in m:
                weight_init(md)
                
            
class OverfitDecoder(torch.nn.Module):
    def __init__(self, input_size=512, out_res=1024):
        super(OverfitDecoder, self).__init__()
        self.input_size = input_size
        self.out_res = out_res
        
        self.ll = nn.Linear(self.input_size, 3*(self.out_res**2))
        self.lr = nn.LeakyReLU()
        self.polish = nn.Conv2d(3, 3, 5, padding=2)
        
        init_model_weights(self)
        
        
    def forward(self, x):
        x = self.ll(x)
        x = x.view(-1, 3, self.out_res, self.out_res)
        #x = self.lr(x)
        #x = self.polish(x)
        return x

                
                
class DecoderV2(torch.nn.Module):
    def __init__(self, input_size=512, start_res=32, out_res=1024):
        super(DecoderV2, self).__init__()
        
        self.input_size = input_size
        self.start_res = start_res
        self.out_res = out_res
        self.LRELU = nn.LeakyReLU()
        
        self.ups = nn.ModuleList([])
        
        self.ll = nn.Linear(self.input_size, self.input_size*(self.start_res**2))
        self.bn1 = nn.BatchNorm2d(self.input_size)
        
        num_double = int(log(self.out_res/self.start_res, 2))
        cur_channels = input_size
        down_factor = max((3/cur_channels)**(-1./num_double), 1) if num_double > 1 else 1
        print('Down factor: {}'.format(down_factor))
        for i in range(num_double):
            self.ups.append(nn.LeakyReLU())
            self.ups.append(nn.ConvTranspose2d(cur_channels, round(cur_channels/down_factor), 2, stride=2))
            cur_channels = round(cur_channels/down_factor)
            #self.ups.append(nn.BatchNorm2d(cur_channels))
            #self.ups.append(nn.Conv2d(cur_channels, cur_channels, 5, padding=2))
        
        
        init_model_weights(self)
        
    def forward(self, x):
        x = self.ll(x)
        #x = self.LRELU(x)
        x = x.view(-1, self.input_size, self.start_res, self.start_res)
        #x = self.bn1(x)
        #x = x.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, self.start_res, self.start_res)
        for u in self.ups:
            x = u(x)
        x = nn.functional.interpolate(x, self.out_res)
        
        return x

class DecoderV1(torch.nn.Module):
    def __init__(self, input_size=512, start_channels=32, out_res = 1024, linear_expansion_factor=1, resolution_expansion_factor=2):
        super(DecoderV1, self).__init__()
        
        self.input_size = input_size
        self.start_res = int(sqrt(self.input_size / start_channels)) * resolution_expansion_factor
        self.start_channels = start_channels * linear_expansion_factor
        print('Starting transposed convolutions with input of size ({}, {}, {})'.format(self.start_channels, self.start_res, self.start_res))
        self.out_res = out_res
        num_double = int(log(self.out_res/self.start_res, 2))
        print('Doubling {} times'.format(num_double))
        
        self.t1 = nn.Linear(self.input_size, self.start_channels*(self.start_res**2))
        self.LRELU = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm2d(self.start_channels)
        
        self.ups = nn.ModuleList([])
        cur_channels = self.start_channels
        down_factor = max((3/cur_channels)**(-1./num_double), 1)
        print('Down factor: {}'.format(down_factor))
        for i in range(num_double):
            self.ups.append(nn.LeakyReLU())
            self.ups.append(nn.ConvTranspose2d(cur_channels, round(cur_channels/down_factor), 2, stride=2))
            cur_channels = round(cur_channels/down_factor)
            #self.ups.append(nn.BatchNorm2d(cur_channels))
            """
            if i % 2 == 0:
                self.ups.append(nn.ConvTranspose2d(cur_channels, cur_channels-4, 2, stride=2))
                cur_channels -= 4
            else:
                self.ups.append(nn.ConvTranspose2d(cur_channels, cur_channels-2, 2, stride=2))
                cur_channels -= 2
            """    
        #self.polish = nn.ConvTranspose2d(cur_channels, 3, 1, stride=1)
        
        
    def forward(self, x):
        x = self.t1(x)
        x = x.view(-1, self.start_channels, self.start_res, self.start_res)
        x = self.LRELU(x)
        x = self.bn1(x)
        for u in self.ups:
            x = u(x)
        #x = self.polish(x)
        x = nn.functional.interpolate(x, self.out_res)
        #x = torch.sigmoid(x)
        return x
    
    
if __name__ == '__main__':
    ips = 512
    #d = DecoderV2(input_size = ips, out_res=1024)
    d = OverfitDecoder(input_size = ips, out_res=128)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # PyTorch v0.4.0
    d.to(device)
    summary(d, (ips,), batch_size=32)
