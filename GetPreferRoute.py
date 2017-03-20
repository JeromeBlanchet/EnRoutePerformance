# -*- coding: utf-8 -*-
"""
Created on Tue Jun 07 13:57:20 2016

@author: Yulin
"""

import numpy as np
import csv
import matplotlib.pyplot as plt

class PreferRoute:
    """
    test code:
        #LALA = PreferRoute()
        #LALA.VisualRoute('FLL','JFK','CD')
    """
    def __init__(self,FAP = 'US_Core_Airport.csv',FVOR='Coords_VOR.csv',FWAY='Coords_Waypnt2.csv',FPREF='PreferredRoute34Core.csv',FCD = 'CodedDepatureRoute34Core.csv'):
        self.FAP = FAP        
        self.FVOR = FVOR
        self.FWAY = FWAY
        self.FPREF = FPREF  
        self.FCD = FCD
        self.CoreAir = self.GetCoreAir()
        self.NavAid = self.GetNaviaid()
        self.Pref_Route = self.GetPreferRoute()
        self.CD_Route = self.GetCDRoute()
    
    def GetCoreAir(self):
        CoreAir = {}
        with open(self.FAP,'r') as csvfile:
            line = csv.reader(csvfile)
            for row in line:
                CoreAir['K'+row[0]] = [float(row[1]),float(row[2])]
        return CoreAir
    def GetNaviaid(self):
        NavAid = {}
        with open(self.FVOR,'r') as csvfile:
            line = csv.reader(csvfile)
            next(line)
            for row in line:
                NavAid[row[0]] = [float(row[2]),float(row[3])]
        with open(self.FWAY,'r') as csvfile2:
            line2 = csv.reader(csvfile2)
            for row in line2:
                NavAid[row[0]] = [float(row[1]),float(row[2])]
        return NavAid
    
    def GetPreferRoute(self):
        Pref_Route = {}
        keylist = []
        with open(self.FPREF,'r') as csvfile:
            line = csv.reader(csvfile)
            for row in line:
                key = row[0] + '_' + row[2]
                CoordList = []
                CoordList.append(self.CoreAir['K'+row[0]])
                Route = row[1].split(' ')
                for item in Route:
                    if self.NavAid.has_key(item):
                        CoordList.append(self.NavAid[item])
                    else:
                        pass
                CoordList.append(self.CoreAir['K'+row[2]])
                if key not in keylist:
                    i = 0
                    Pref_Route[key] = {}
                    keylist.append(key)
                else:
                    i += 1
                    pass
                Pref_Route[key][str(i)] = np.asarray(CoordList)
                
        return Pref_Route
        
    def GetCDRoute(self):
        CD_Route = {}
        keylist = []
        with open(self.FCD,'r') as csvfile:
            line = csv.reader(csvfile)
            for row in line:
                key = row[1] + '_' + row[2]
                CoordList = []
                CoordList.append(self.CoreAir[row[1]])
                Route = row[4].split(' ')
                for item in Route:
                    if self.NavAid.has_key(item):
                        CoordList.append(self.NavAid[item])
                    else:
                        pass
                CoordList.append(self.CoreAir[row[2]]) 
                if key not in keylist:
                    i = 0
                    CD_Route[key] = {}
                    keylist.append(key)
                else:
                    i += 1
                    pass
                CD_Route[key][str(i)] = np.asarray(CoordList)
                
        return CD_Route
    
    def VisualRoute(self, Ori, Des, TYPE):
        key = 'K'+Ori + '_K' + Des
        if TYPE == 'PR':
            for i in range(len(self.Pref_Route[key])):
                print( i )
                ax = plt.plot(self.Pref_Route[key][str(i)][:,1],self.Pref_Route[key][str(i)][:,0],'o-')
            return ax
        elif TYPE == 'CD':
            for i in range(len(self.CD_Route[key])):
                print( i )
                ax = plt.plot(self.CD_Route[key][str(i)][:,1],self.CD_Route[key][str(i)][:,0],'*-')
            return ax
        else:
            raise ValueError('WRONG TYPE!')
