# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 22:53:09 2024

@author: adrie
"""

import torch
import torch.nn as nn

class ModelDraft(torch.nn.Module):
    def __init__(self,dim_champ,num_champ,dim_comp):
        super().__init__()
        #Embeddings
        self.echamp=torch.nn.Embedding(num_champ, dim_champ)
        dim_element=dim_champ
        
        self.dense_team_comp=torch.nn.Linear(dim_element*5,dim_comp)
        
        self.out=torch.nn.Linear(dim_comp*2,1)
        self.a0=torch.nn.ReLU(inplace=True)
        self.d=torch.nn.Dropout(0.2)
        self.dense1=torch.nn.Linear(dim_comp*2,16*2)
    def forward(self,x):
        bs=x["champion1"].shape[0]
        e1=self.echamp(x["champion1"])#.mean(axis=-2)
        e2=self.echamp(x["champion2"])#.mean(axis=-2)
        e1=self.d(e1)
        e2=self.d(e2)
        
        e1=self.dense_team_comp(e1.reshape(bs,-1))
        e2=self.dense_team_comp(e2.reshape(bs,-1))
        e=torch.cat([e1,e2],axis=-1)
        e=e.reshape(bs,-1)
        e=self.a0(e)
        e=self.d(e)
        e=self.out(e)
        return e

class ModelDraft(torch.nn.Module):
    def __init__(self,dim_champ,num_champ,dim_comp):
        super().__init__()
        #Embeddings
        self.echamp=torch.nn.Embedding(num_champ, dim_champ)
        dim_element=dim_champ
        
        self.dense_team_comp=torch.nn.Linear(dim_element*5,dim_comp)
        
        self.out=torch.nn.Linear(dim_comp*2,1)
        self.a0=torch.nn.ReLU(inplace=True)
        self.d=torch.nn.Dropout(0.2)
        self.dense1=torch.nn.Linear(dim_comp*2,16*2)
        self.positionnal_embedding=torch.nn.Embedding(5, dim_champ)

    def forward(self,x):
        bs=x["champion1"].shape[0]
        e1=self.echamp(x["champion1"])#.mean(axis=-2)
        e2=self.echamp(x["champion2"])#.mean(axis=-2)
        p=self.positionnal_embedding(torch.arange(0,5,1))
        
        e1=e1+p[None,:,:]
        e2=e2+p[None,:,:]
        
        e1=self.d(e1)
        e2=self.d(e2)
        
        e1=self.dense_team_comp(e1.reshape(bs,-1))
        #e1=self.a0(e1)
        #e1=self.dense_team_comp2(e1)
        
        e2=self.dense_team_comp(e2.reshape(bs,-1))
        #e2=self.a0(e2)
        #e2=self.dense_team_comp2(e2.reshape(bs,-1))
        
        #e=self.draft_attention(e2,e1)
        e=torch.cat([e1,e2],axis=-1)
        #e=self.dense1(e)
        e=e.reshape(bs,-1)
        e=self.a0(e)
        e=self.d(e)
        #e=self.dense2(e)
        #e=self.a0(e)
        #e=self.dense3(e)
        #e=self.a0(e)
        #e=self.dense4(e)
        #e=self.a0(e)
        e=self.out(e)
        
        
        return e

class ModelDraft(nn.Module):
    def __init__(self):
        super(ModelDraft, self).__init__()
        dim_champ=1024
        new_dim_champ=128
        dim_comp=64
        hidden_size=dim_comp*2
        num_champ=168
        self.echamp=torch.nn.Embedding(num_champ, dim_champ)
        self.base_layer=nn.Sequential(nn.Linear(dim_champ,new_dim_champ),
                                      nn.ReLU(inplace=True))
        self.shared_layer=nn.Sequential(nn.Linear(new_dim_champ*5,new_dim_champ*5),
                                      nn.ReLU(inplace=True))
        
        self.positionnal_embedding=torch.nn.Embedding(5, new_dim_champ)
        self.dense_team_comp=torch.nn.Linear(new_dim_champ*5,dim_comp)
        #self.dense_team_comp2=torch.nn.Linear(dim_comp,dim_comp)
        #self.dense2=torch.nn.Linear(128*2,64*2)
        self.out=torch.nn.Linear(32,1)
        self.a0=torch.nn.ReLU(inplace=True)
        self.a00=torch.nn.ReLU(inplace=True)
        self.d=torch.nn.Dropout(0.2)
        #self.cross_dense=nn.Linear(5*dim_champ,5*64)
        self.mse_head = nn.Linear(2*new_dim_champ, 1)
        #self.mse_head = nn.Linear(hidden_size, 1)  # Output for MSE
        self.bce_head = nn.Sequential(nn.Linear(hidden_size,hidden_size),
                                      nn.ReLU(inplace=True),
                                       nn.Linear(hidden_size, 1))  # Output for BCE

    def forward(self, x):
        bs=x["champion1"].shape[0]
        e1=self.echamp(x["champion1"])#.mean(axis=-2)
        e2=self.echamp(x["champion2"])#.mean(axis=-2)
        e1=self.base_layer(e1)
        e2=self.base_layer(e2)
        p=self.positionnal_embedding(torch.arange(0,5,1))
        e1=e1+p[None,:,:]
        e2=e2+p[None,:,:]
        
        e1=self.shared_layer(e1.reshape(bs,-1)).reshape(bs,5,-1)
        e2=self.shared_layer(e2.reshape(bs,-1)).reshape(bs,5,-1)
        e1=self.d(e1)
        e2=self.d(e2)
        #GP=torch.cat([self.cross_dense(e1.reshape(bs,-1)).reshape(bs,5,-1),self.cross_dense(e2.reshape(bs,-1)).reshape(bs,5,-1)],axis=-1)
        GP=torch.cat([e1,e2],axis=-1)
        e1=self.dense_team_comp(e1.reshape(bs,-1))
        e2=self.dense_team_comp(e2.reshape(bs,-1))
        e=torch.cat([e1,e2],axis=-1)
        e=self.a0(e)
        # Shared layers
        #x = self.shared_layers(x)
        
        # MSE head
        mse_output = self.mse_head(GP)
        
        # BCE head
        bce_output = self.bce_head(e)

        return bce_output,mse_output

