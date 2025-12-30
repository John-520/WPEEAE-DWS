# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 19:12:18 2021

@author: 29792
"""
import numpy as np
import torch
import torch.nn as nn
from loss.DAN import DAN
from loss.MMD import mmd_rbf_noaccelerate, mmd_rbf_accelerate
from loss.JAN import JAN
from loss.CORAL import CORAL
import sys 
sys.path.append("D:\北京交通大学博士\论文【小】\论文【第四章】\code") 
from MMSD_main.MMSD import MMSD

from loss.lmmd import LMMD_loss
from loss.contrastive_center_loss import ContrastiveCenterLoss
from loss.SupervisedContrastiveLoss import SupervisedContrastiveLoss
from loss.ContrastiveLoss import ContrastiveLoss

from loss.SupConLoss import SupConLoss

import sub_models
import torch.nn.functional as F

from sklearn.cluster import KMeans

from transfer_losses import TransferLoss
from loss.adv import *
import CKButils


from timm.loss import LabelSmoothingCrossEntropy

from loss.DANCE_loss import *


from torch.autograd import Variable
import matplotlib.pyplot as plt

import ot
from scipy.optimize import linear_sum_assignment

import torch.nn.init as init

def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    
    
# compute entropy loss
def get_entropy_loss(p_softmax):
    mask = p_softmax.ge(0.00000001)
    mask_out = torch.masked_select(p_softmax, mask)
    entropy = -(torch.sum(mask_out * torch.log(mask_out)))
    
    return 0.1 * (entropy / float(p_softmax.size(0)))


# compute entropy
def HLoss(x):
    b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
    b = -1.0 * b.sum()
    return b





'行列正则化'
def l2row_torch(X):
	"""
	L2 normalize X by rows. We also use this to normalize by column with l2row(X.T)
	"""
	N = torch.sqrt((X**2).sum(axis=1)+1e-8)
	Y = (X.T/N).T
	return Y,N
 
        






class models(nn.Module):

    def __init__(self, args):
        super(models, self).__init__()
        self.args = args
        
        self.num_classes= args.num_classes
        self.adversarial_loss = self.args.adversarial_loss
        #这'句话很重要'
        
        
        
        self.feature_layers = getattr(sub_models, args.model_name)(args, args.pretrained)
        
        
        self.ad_layers = getattr(sub_models, 'DomainClassifier')(100,1)

        self.bottle = nn.Sequential(nn.Linear(768, 150),  # 192    384
                                    nn.GELU(), 
                                    nn.Dropout()
                                    ) #
        self.bottle_taskA = nn.Sequential(nn.Linear(self.feature_layers.output_num(), 256),
                                    nn.GELU(), 
                                    nn.Dropout()
                                    ) #
        
        
        self.drop = nn.Dropout(0)
        if self.args.model_name == 'RFFDN':
            self.cls_fc = nn.Linear(self.feature_layers.output_num(), self.num_classes)
        else:
            self.cls_fc = nn.Linear(150, self.num_classes) #
            
            self.cls_fc_taskA = nn.Linear(256, 3)     # 多任务下的分类器  ；或者是辅助任务下的分类器
            self.cls_fc_taskB = nn.Linear(256, 2) 
            self.cls_fc_taskC = nn.Linear(256, 2) 
            
            
            self.cls_fc1 = nn.Linear(self.feature_layers.output_num(), 256) #
            self.cls_fc2 = nn.Linear(256, self.num_classes) #
            
            self.cls_fc_pseudo = nn.Linear(1, self.num_classes) #
        #定义对比损失函数的输入
        self.prelu_ip1 = nn.PReLU()
        self.contrast = nn.Linear(self.feature_layers.output_num(), 2)  #
        
        
        self.Sigmoid = nn.Sigmoid()
        self.SwiGLU = nn.GELU()
        self.Dropout = nn.Dropout()   # 0.5
        
        
        
        
        
        
        
        self.fc_mu = nn.Linear(384, 384 * 2)
        
        
        
        self.bn=nn.BatchNorm1d(384)
        self.head = nn.Sequential(
                nn.Linear(384, 100),
                nn.ReLU(inplace=True),
                nn.Linear(100, 30)
            )
        
        self.PEC_bottle = nn.Sequential(
                nn.Linear(384, self.num_classes),
            )
        
        self.ConvAutoencoder_encoder = getattr(sub_models, 'ConvAutoencoder_encoder')()
        self.ConvAutoencoder_decoder = getattr(sub_models, 'ConvAutoencoder_decoder')()
        
        
        self.ConvAutoencoderVAE_encoder = getattr(sub_models, 'ConvAutoencoderVAE_encoder')()
        self.ConvAutoencoderVAE_decoder = getattr(sub_models, 'ConvAutoencoderVAE_decoder')()
        
        self.ConvAutoencoder_encoder_PCE = getattr(sub_models, 'ConvAutoencoder_encoder_PCE')()
        self.ConvAutoencoder_decoder_PCE = getattr(sub_models, 'ConvAutoencoder_decoder_PCE')()
        
        self.log_vars = nn.Parameter(torch.randn(1))

        
        
        self.log_vars1 = nn.Parameter(torch.randn(1))


        
        
        
        
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            warnings.warn("gpu is not available")
            self.device = torch.device("cpu")
        
        
        
        if args.distance_metric:
            if args.distance_loss == 'MK-MMD':
                self.distance_loss = DAN
            elif args.distance_loss == 'MMD':
                self.distance_loss = mmd_rbf_accelerate  #  mmd_rbf_accelerate   mmd_rbf_noaccelerate
            elif args.distance_loss == 'CKB':
                self.distance_loss = mmd_rbf_accelerate  #  mmd_rbf_accelerate   mmd_rbf_noaccelerate
            elif args.distance_loss == 'CKB+MMD':
                self.distance_loss = mmd_rbf_accelerate  #  mmd_rbf_accelerate   mmd_rbf_noaccelerate
            elif args.distance_loss == "JMMD":
                ## add additional network for some methods
                self.softmax_layer = nn.Softmax(dim=1)
                self.softmax_layer = self.softmax_layer.to(self.device)
                self.distance_loss = JAN
            elif args.distance_loss == "CORAL":
                self.distance_loss = CORAL
            elif args.distance_loss == "MMSD":
                self.distance_loss = MMSD()
                
            elif args.distance_loss == "LMMD":
                self.distance_loss = LMMD_loss(class_num=self.num_classes).to(self.device)
            else:
                raise Exception("loss not implement")
        else:
            self.distance_loss = None
            
            
            
        # Define the contrast loss
        if args.constract_loss_metric:
            if args.constract_loss == 'constract_center_loss':
                self.constract_loss = ContrastiveCenterLoss(dim_hidden=2, num_classes=args.num_classes,
                                            use_cuda=False)
            elif args.constract_loss == 'SupervisedContrastiveLoss':
                self.constract_loss = SupervisedContrastiveLoss(temperature=0.07)
            elif args.constract_loss == 'constract_loss':
                self.constract_loss = ContrastiveLoss(margin=2.27)
            else:
                raise Exception("loss not implement")
        else:
            self.constract_loss = None



        #定义对抗损失函数
        transfer_loss_args = {
            "loss_type": self.args.adversarial_loss,
        }
        self.adapt_loss = TransferLoss(**transfer_loss_args)


    def l2_zheng(self,x):
        x = torch.sqrt(x**2 + 1e-8)
        x = l2row_torch(x)[0]
        x = l2row_torch(x.T)[0].T        
        return x


    def forward(self, source,target,label_source,label_target, index_target, epoch,mu_value,task, ndata): #(64,1,1200)
        loss = 0
        const_loss = 0
        dist_loss =0
        adv_loss=0
        dist_loss_shang = 0
        dist_loss_KB = 0
        
        
        dist_loss_mmd , dist_loss_jmmd,  dist_loss_mmsd, dist_loss_lmmd, dist_loss_coral = 0, 0, 0, 0, 0
        dist_loss_kmmd = 0
        
        
        
        source_source = source[0]["hello"].cuda()
        target_target = target[0]["hello"].cuda()
        source_PCE = source[3]["hello"].squeeze().cuda()
        

        s_pred_A = 0

        
        
        ############################################################################################################独立的自编码
        source_z = self.ConvAutoencoderVAE_encoder(source_source)
        s_z = source_z
        source_z_PCE = self.ConvAutoencoder_encoder_PCE(source_PCE) 
        
        
        source_z = source_z.view(source_z.size(0), -1)
        source_qian = source_z[:, :source_z_PCE.size(1)]
        source_hou =  source_z[:, -source_z_PCE.size(1):]
        log_vars_softplus = torch.nn.functional.softplus(self.log_vars1)
        roation = 1- log_vars_softplus / (1 + log_vars_softplus)

        
        source_qian = source_qian * roation
        source_hou = source_hou * (1-roation)
        source_z=torch.cat([source_qian,source_hou],dim=1)
        source_z = source_z.reshape(s_z.shape[0],s_z.shape[1],s_z.shape[2])
        
        
        source_out = self.ConvAutoencoderVAE_decoder(source_z) 
        loss_R1=F.mse_loss(source_source,source_out)
        
        
        source_out_PCE = self.ConvAutoencoder_decoder_PCE(source_z_PCE)
        loss_R2=F.mse_loss(source_PCE,source_out_PCE)
        
        source_z = source_z.view(source_z.size(0), -1)      


        source_PCE = source_PCE.unsqueeze(1)
        source_PCE = source_PCE.repeat(1, 20, 1)
        source_PCE = source_PCE.reshape(source_PCE.shape[0], -1)
        source_PCE = source_PCE[:, :384]


        
        loss_phy=F.mse_loss(source_hou,source_z_PCE)


        if mu_value == 2:    
            print(self.log_vars,roation)
        
        
        

        '分类功能' ############################################################################################################
        source_z = self.l2_zheng(source_z)
        source = self.bottle(source_z)
        source = self.l2_zheng(source)
        s_pred = self.cls_fc(source)
        
        
        PCE_pred = self.PEC_bottle(source_z_PCE)
        loss_cls = F.nll_loss(F.log_softmax(PCE_pred, dim=1), label_source)
        
        
        if self.training == True:    
            
            if epoch< self.args.middle_epoch:
                loss_cls = 0
                loss_PCE = 0
                loss_PCEC = 0
                
                
            loss = loss_R1 + loss_R2  + loss_cls  + loss_phy 
        
        
        
        
        '目标域输入'############################################################################################################
        target = self.ConvAutoencoder_encoder(target_target)
        target = target.view(target.size(0), -1)
    
        target = self.bottle(source_z)
        t_pred = self.cls_fc(target)

        
        
            
        return source_out, source_source, s_pred, s_pred_A, t_pred,   loss,  0 * adv_loss 




    def predict(self, x):
        target_target = x[0]["hello"].cuda()

        
        x = self.ConvAutoencoderVAE_encoder(target_target) 
        x = x.view(x.size(0), -1)
        x = self.l2_zheng(x)
        x = self.bottle(x)
        x = self.l2_zheng(x)

        return self.cls_fc(x)
    

