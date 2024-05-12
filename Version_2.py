# -*- coding: utf-8 -*-
"""
Created on Sat Dec 23 10:48:58 2023

@author: luzy1
"""
import torch
import torch.nn as nn

class ConvAutoencoder_encoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder_encoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(1600, 800),
            nn.ReLU(True),
            nn.Linear(800, 400),
            nn.ReLU(True),
            nn.Linear(400, 200),
            nn.ReLU(True),
            nn.Linear(200, 100),
            nn.ReLU(True),
            nn.Linear(100, 50)
        )

    def forward(self, x):
        x = self.encoder(x)
        return x
    



class ConvAutoencoder_decoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder_decoder, self).__init__()


        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(50, 100),
            nn.ReLU(True),
            nn.Linear(100, 200),
            nn.ReLU(True),
            nn.Linear(200, 400),
            nn.ReLU(True),
            nn.Linear(400, 800),
            nn.ReLU(True),
            nn.Linear(800, 1600)
        )

    def forward(self, x):
        x = self.decoder(x)
        return x
    
    
    
    
    
    
    
    
    
    
    
    
    
