# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 22:12:28 2024

@author: adrie
"""

import json
import pathlib
import os

class ChampionParser:
    def __init__(self):
        path_current_file=os.path.abspath(pathlib.Path(__file__).parent.resolve())
        self.path=path_current_file+"/Champion_Data/"
        self.file_description="champion_description.json"
        self.get_description()
        self.get_icon()
        pass
    
    def get_description(self):
        path=os.path.normpath(self.path+self.file_description)
        with open(path, "r") as f:
            file=f.read()
        self.dic_champ={}
        for dic in json.loads(file):
            if dic["name"]!="None":
                self.dic_champ[dic["name"]]=dic

    def get_icon(self):
        for i in self.dic_champ:
            self.dic_champ[i]['icon']=self.path+"icons/"+str(self.dic_champ[i]["id"])+".png"
        pass

    def get_list_name(self):
        return list(self.dic_champ.keys())

