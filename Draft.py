# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 16:57:44 2024

@author: adrie
"""
from tkinter import Tk, Button, Entry, Label
from PIL import ImageTk, Image
from ChampionParser import ChampionParser
import os
import pathlib
import torch
from Model.Model import ModelDraft
import joblib
import numpy as np
cur_path=os. path. abspath(os.curdir)+"/"
abs_path=os.path.abspath(pathlib.Path(__file__).parent.resolve())+"/"


CP=ChampionParser()
class Prediction:
    def __init__(self):
        encoder_filename = "champion_encoder.save"
        self.encoder = joblib.load(abs_path+"Model/"+encoder_filename)
        self.model=ModelDraft()  #dim_champ=1024,num_champ=168,dim_comp=16
        self.model.load_state_dict(torch.load('Model/draft55_seq.pth'))
        self.model.eval()
    
    def predict(self,draft):
        dic={}
        dic["champion1"]=draft[:5]#["Tristana","Poppy","Taliyah","Lucian","Nami"]
        dic["champion2"]=draft[5:]#["Twisted Fate","Rell","Ahri","Kalista","Renata Glasc"]
        dic["champion1"]=self.encoder.transform(dic["champion1"])
        dic["champion2"]=self.encoder.transform(dic["champion2"])
        dic={k:torch.LongTensor(dic[k]).reshape(1,-1) for k in dic}
        a,b=self.model(dic)
        a=a.item()
        b=b.detach().numpy()
        b=b[0].T[0]
        x=" ".join([str(np.round(i,3))[:5] for i in b])
        return str(np.round(1/(1+np.exp(-a)),3)*100)[:4]+" %"

class champion:
    def __init__(self,name="Azyr",root=None):
        self.panel=None
        self.button=None
        self.name=name
        self.root=root
        self.Parser=CP
        
    def open_image(self,col,row,C,i):
        img = Image.open(self.Parser.dic_champ[self.name]["icon"]).resize((64,64))
        img_=ImageTk.PhotoImage(img)#.resize((128,128))
        self.close_image()
            
        self.panel = Label(self.root, image = img_)
        self.panel.image=img_
        self.panel.grid(column=col+1,row=row)#pack(side = "bottom", fill = "both", expand = "yes")
        
        self.button=Button(self.root, text="Delete")
        self.button.grid(column=col+1,row=row+1)#pack()
        def destroy():
            self.button.destroy()
            self.close_image()
            C[i]=None
        self.button['command'] = destroy
        
    def close_image(self):
        if self.panel:
            self.panel.destroy()
        if self.button:
            self.button.destroy()

class MyWindow(Tk):

    def __init__(self):
        # On appelle le constructeur parent
        super().__init__()
        
        self.Parser=CP
        self.Predictor=Prediction()
        
        self.post_list=["Top","Jungle","Mid","ADC","Sup"]
        self.l=None
        
        
        for i in range(12):
            self.grid_rowconfigure(i, weight=1)
        for i in range(5):
            self.grid_columnconfigure(i, weight=1)
        B=[]
        E=[]
        self.B=B
        self.E=E
        self.C=[]
        for i in range(0,5):
            self.B += [Button(self, text=self.post_list[i])]
            self.E+=[Entry(self)]
            col=0
            row=i*2
            self.B[-1].grid(column=col, row=row)
            self.B[-1]["command"]=lambda col=col,row=row,i=i :self.get_champ(col,row,i)
            self.E[-1].grid(column=col, row=row+1)
            self.C+=[None]
        
        for i in range(5,10):
            self.B += [Button(self, text=self.post_list[i-5])]
            self.E+=[Entry(self)]
            col=3
            row=(i-5)*2
            self.B[-1].grid(column=col, row=row)
            self.B[-1]["command"]=lambda col=col,row=row,i=i :self.get_champ(col,row,i)
            self.E[-1].grid(column=col, row=row+1)
            self.C+=[None]
        # On dimensionne la fenêtre (400 pixels de large par 400 de haut).
        self.geometry("600x600")

        # On ajoute un titre à la fenêtre
        self.title("Draft")
        self.main_button=Button(self, text="Prediction",command=self.get_prediction)
        self.main_button.grid(row=10,column=2)
        
    def get_prediction(self):
        if self.l:
            self.l.destroy()
        if None in self.C:
            self.l=Label(text="It's a 5v5, champion missing")
            self.l.grid(row=11,column=2)
        else:
            draft=[C.name for C in self.C]
            pred=self.Predictor.predict(draft)
            self.l=Label(text=f"{pred} % de chance de victoire équipe de gauche")
            self.l.grid(row=11,column=2)
            
            
    def get_champ(self,col,row,i):
        name=self.E[i].get()
        if self.C[i]:
            self.C[i].close_image()
        if name in self.Parser.get_list_name():
            C=champion(root=self,name=name)
        else :
            C=champion(root=self,name="Annie")
        C.open_image(col,row,self.C,i)
        self.C[i]=C
# On crée notre fenêtre et on l'affiche

window = MyWindow()
window.mainloop()