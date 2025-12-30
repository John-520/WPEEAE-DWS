# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 16:36:11 2022

@author: 29792
"""

# import torch
# from torch import nn
# from torch.nn import functional as F



# class ChannelAttention(nn.Module):
#     def __init__(self, in_planes, ratio=16):
#         super(ChannelAttention, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool1d(1)
#         self.max_pool = nn.AdaptiveMaxPool1d(1)
#         self.shared_MLP = nn.Sequential(
#             nn.Conv1d(in_planes, in_planes // ratio, 1, bias=False),
#             nn.ReLU(),
#             nn.Conv1d(in_planes // ratio, in_planes, 1, bias=False)
#         )
#         # self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
#         # self.relu1 = nn.ReLU()
#         # self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         avg_out =self.shared_MLP(self.avg_pool(x))# self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
#         max_out =self.shared_MLP(self.max_pool(x))# self.fc2(self.relu1(self.fc1(self.max_pool(x))))
#         out = avg_out + max_out
#         return self.sigmoid(out)


# class SpatialAttention(nn.Module):
#     def __init__(self, kernel_size=7):
#         super(SpatialAttention, self).__init__()

#         assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
#         padding = 3 if kernel_size == 7 else 1

#         self.conv1 = nn.Conv1d(2, 1, kernel_size, padding=padding, bias=False)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         avg_out = torch.mean(x, dim=1, keepdim=True)
#         max_out, _ = torch.max(x, dim=1, keepdim=True)
#         x = torch.cat([avg_out, max_out], dim=1)
#         x = self.conv1(x)
#         return self.sigmoid(x)


# class CBAM(nn.Module):
#     def __init__(self, planes):
#         super(CBAM, self).__init__()
#         self.ca = ChannelAttention(planes)
#         self.sa = SpatialAttention()

#     def forward(self, x):
#         x = self.ca(x) * x
#         x = self.sa(x) * x
#         return x

# if __name__ == '__main__':
#     img = torch.randn(16, 1200, 20)
#     net = CBAM(1200)
#     # print(net)
#     out = net(img)
#     print(out.size())
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

import torch
import torch.nn as nn
class CBAMLayer(nn.Module):
    def __init__(self, channel, reduction=16, spatial_kernel=7):
        super(CBAMLayer, self).__init__()
        # channel attention 压缩H,W为1
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        # shared MLP
        self.mlp = nn.Sequential(
            # Conv2d比Linear方便操作
            # nn.Linear(channel, channel // reduction, bias=False)
            nn.Conv1d(channel, channel // reduction, 1, bias=False),
            # inplace=True直接替换，节省内存
            nn.ReLU(inplace=True),
            # nn.Linear(channel // reduction, channel,bias=False)
            nn.Conv1d(channel // reduction, channel, 1, bias=False)
        )
        # spatial attention
        self.conv = nn.Conv1d(2, 1, kernel_size=spatial_kernel,
                              padding=spatial_kernel // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        max_out = self.mlp(self.max_pool(x))
        avg_out = self.mlp(self.avg_pool(x))
        channel_out = self.sigmoid(max_out + avg_out)
        x = channel_out * x
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        spatial_out = self.sigmoid(self.conv(torch.cat([max_out, avg_out], dim=1)))
        x = spatial_out * x
        return x
    
x = torch.randn(1,1200,32)
net = CBAMLayer(1200)
y = net.forward(x)
print(y.shape)


