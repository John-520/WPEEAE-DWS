# -*- coding: utf-8 -*-
"""
Created on Sat Dec 23 10:48:58 2023

@author: luzy1
"""
import torch
from torch import nn
import torch.nn.functional as F
class ConvAutoencoder_encoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder_encoder, self).__init__()

        #编码器
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=81, stride=8, padding=1, dilation=1),
            nn.BatchNorm1d(16),
            nn.ReLU(True),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),

            nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(True),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(True),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(True),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(True),
            # nn.MaxPool1d(kernel_size=2, stride=2),
        )


        #潜在空间的维度
        self.latent_dim = 384
        #均值和方差的全连接层
        self.fc_mu = nn.Linear(384, 384 * 2)
        
        
        self.fc_logvar = nn.Linear(384, self.latent_dim)

    def reparameterize(self, mu, logvar):
        #计算标准差
        std = torch.exp(0.5 * logvar)
        #从标准正态分布中采样
        eps = torch.randn_like(std)
        #根据公式得到潜在变量
        z = mu + eps * std
        return z
    
    
    def forward(self, x):
        x = self.encoder(x)
        # x = x.view(x.size(0), -1)
        # x = self.fc_mu(x)
        s_z = x
        #将输出展平
        # x = x.view(x.size(0), -1)
        #计算均值和方差
        # mu = self.fc_mu(x)
        # logvar = self.fc_logvar(x)
        
        #根据均值和方差进行重参数化，得到潜在变量
        # z = self.reparameterize(mu, logvar)
        
        return s_z #, z, mu, logvar
    
    

    
        #编码器2
    #     self.conv1 = nn.Conv1d(1, 32, kernel_size=3, stride=1, padding=1)
    #     self.bn1 = nn.BatchNorm1d(32)
    #     self.conv2 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)
    #     self.bn2 = nn.BatchNorm1d(64)
    #     self.conv3 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1)
    #     self.bn3 = nn.BatchNorm1d(128)
    #     self.conv4 = nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1)
    #     self.bn4 = nn.BatchNorm1d(128)
    #     self.conv5 = nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1)
    #     self.bn5 = nn.BatchNorm1d(256)
    #     self.conv6 = nn.Conv1d(256, 512, kernel_size=3, stride=1, padding=1)
    #     self.bn6 = nn.BatchNorm1d(512)

    #     self.pool = nn.MaxPool1d(kernel_size=2, stride=2)


    # def forward(self, x):
    #     # Input shape: (batch_size, 1, input_size)
    #     x = self.conv1(x)
    #     x = self.bn1(x)
    #     x = nn.functional.relu(x)
    #     x = self.pool(x)

    #     x = self.conv2(x)
    #     x = self.bn2(x)
    #     x = nn.functional.relu(x)
    #     x = self.pool(x)

    #     x = self.conv3(x)
    #     x = self.bn3(x)
    #     x = nn.functional.relu(x)
    #     x = self.pool(x)

    #     x = self.conv4(x)
    #     x = self.bn4(x)
    #     x = nn.functional.relu(x)
    #     x = self.pool(x)

    #     x = self.conv5(x)
    #     x = self.bn5(x)
    #     x = nn.functional.relu(x)
    #     x = self.pool(x)

    #     x = self.conv6(x)
    #     x = self.bn6(x)
    #     x = nn.functional.relu(x)
    #     x = self.pool(x)


    #     return x

    
    
    
    
    
    

class ConvAutoencoder_decoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder_decoder, self).__init__()


        #解码器       
        # self.decoder = nn.Sequential(
        #     nn.ConvTranspose1d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
        #     nn.BatchNorm1d(64),
        #     nn.ReLU(True),

        #     nn.ConvTranspose1d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
        #     nn.BatchNorm1d(64),
        #     nn.ReLU(True),

        #     nn.ConvTranspose1d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
        #     nn.BatchNorm1d(64),
        #     nn.ReLU(True),

        #     nn.ConvTranspose1d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
        #     nn.BatchNorm1d(32),
        #     nn.ReLU(True),

        #     nn.ConvTranspose1d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=0),
        #     nn.BatchNorm1d(16),
        #     nn.ReLU(True),

        #     nn.ConvTranspose1d(16, 1, kernel_size=80, stride=8, padding=0, output_padding=0),
        #     # nn.Sigmoid()  # Assuming you want the output in the range [0, 1]
        # )



        #解码器       
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(True),

            nn.ConvTranspose1d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(True),

            nn.ConvTranspose1d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(True),

            nn.ConvTranspose1d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(True),

            nn.ConvTranspose1d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=0),
            nn.BatchNorm1d(16),
            nn.ReLU(True),

            nn.ConvTranspose1d(16, 1, kernel_size=80, stride=4, padding=4, output_padding=0),
            # nn.Sigmoid()  # Assuming you want the output in the range [0, 1]
        )
        
        

    def forward(self, x):
        x = self.decoder(x)
        return x
    
    
    
    
    
    

        # 六层解码器
    #     self.deconv6 = nn.ConvTranspose1d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1)
    #     self.bn6 = nn.BatchNorm1d(256)
    #     self.deconv5 = nn.ConvTranspose1d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
    #     self.bn5 = nn.BatchNorm1d(128)
    #     self.deconv4 = nn.ConvTranspose1d(128, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
    #     self.bn4 = nn.BatchNorm1d(128)
    #     self.deconv3 = nn.ConvTranspose1d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
    #     self.bn3 = nn.BatchNorm1d(64)
    #     self.deconv2 = nn.ConvTranspose1d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
    #     self.bn2 = nn.BatchNorm1d(32)
    #     self.deconv1 = nn.ConvTranspose1d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1)
    #     self.bn1 = nn.BatchNorm1d(1)
        
    #     # self.sigmoid = nn.Sigmoid() 

    # def forward(self, x):
    #     # Input shape: (batch_size, channels, input_size)
       
    #     x = F.relu(self.bn6(self.deconv6(x)))
    #     x = F.relu(self.bn5(self.deconv5(x)))
    #     x = F.relu(self.bn4(self.deconv4(x)))
    #     x = F.relu(self.bn3(self.deconv3(x)))
    #     x = F.relu(self.bn2(self.deconv2(x)))
    #     x = self.deconv1(x)

    #     return x

    
    
    
    
    
    
    
    
    
    
    
    
    
