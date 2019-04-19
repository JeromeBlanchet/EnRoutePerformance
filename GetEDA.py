# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 16:11:02 2016

@ Author: Liu, Yulin
@ Institute: UC Berkeley
"""
from __future__ import division
import numpy as np
import pandas as pd
import csv
import pymongo
from mpl_toolkits.basemap import Basemap
from descartes import PolygonPatch
from shapely.geometry import Polygon, Point
import matplotlib.pyplot as plt
import time
import os
import math
from dateutil.parser import parse
from shapely.geometry import LineString
import pickle
from GetPreferRoute import PreferRoute
import functools

def GCDistance(S_Lat,S_Lon,E_Lat,E_Lon):
    start_lat = np.radians(S_Lat)  
    start_lon = np.radians(S_Lon)
    end_lat = np.radians(E_Lat)
    end_lon = np.radians(E_Lon)
    d_lat = end_lat - start_lat  
    d_lon = end_lon - start_lon  
    a = np.sin(d_lat/2)**2 + np.cos(start_lat) * np.cos(end_lat) * np.sin(d_lon/2)**2  
    c = 2 * np.arcsin(np.sqrt(a))  
    # Radius of earth is 3440 nm
    return 3440 * c

def GetCoords():
    Coords = {}
    with open('US_Core_Airport.csv','r') as csvfile:
        line = csv.reader(csvfile)
        for row in line:
            Coords[row[0]] = [float(row[1]),float(row[2])]
    return Coords
    
class EDA_Data:
    
    def __init__(self, Dep, Arr, year, Cutoff = 0.5, Tcut = 300, Dcut = 100, 
                 Vcut = 0.27, db = False, Insert = False, InputData = False):
        """
        Input Type:
        Dep: String, 
        Arr: String, 
        year: Int, 

        """        
        print('Warning: If there is no existed valid flight track data, you should build up mongodb connection!\n')
        self.Dep = Dep
        self.Arr = Arr        
        self.year = year
        self.Coords = GetCoords()
        self.Ori_Lat = self.Coords[self.Dep][0]
        self.Ori_Lon = self.Coords[self.Dep][1]
        self.Des_Lat = self.Coords[self.Arr][0]
        self.Des_Lon = self.Coords[self.Arr][1]
        
        if InputData:
            try:
                EfficiencyPath = os.getcwd() + '/TFMS_NEW/Eff_' + self.Dep + self.Arr + str(self.year) + '.p'
                self.Efficiency = pickle.load(open(EfficiencyPath,'rb'))
                print('Inefficiency File Loaded')
            except:
                print('Location Guess is Wrong! Please Enter the locations for the Effiency Pickle File!')
                print('Example:\nG:\\00_BERKELEY\\01_RESEARCH\\03_Efficiency\\05_TFMS_Traj\\TFMS_NEW\\Eff_IAHBOS2013.p')
                EfficiencyPath = raw_input()
                self.Efficiency = pickle.load(open(EfficiencyPath,'rb'))
            try:
                VTrackPath = os.getcwd() + '/TFMS_NEW/New_' + self.Dep + self.Arr + str(self.year) + '.csv'
                self.VTrack = pd.read_csv(VTrackPath)
                print('Valid Track File Loaded')
            except:
                print('Location Guess is Wrong! Please Enter the locations for the Valid Flight Track File!')
                print('Example:\nG:\\00_BERKELEY\\01_RESEARCH\\03_Efficiency\\05_TFMS_Traj\\TFMS_NEW\\New_IAHBOS2013.csv')
                VTrackPath = raw_input()
                self.VTrack = pd.read_csv(VTrackPath)
#            try:
#                VFPPath = os.getcwd() + '/FlightPlans/FlightPlans_' + self.Dep + '_' + self.Arr + '_' + str(self.year) + 'FP_PT.csv'
#                self.VFP = pd.read_csv(VFPPath)
#                print('Valid Flight Plan File Loaded')
#            except:
#                print('Location Guess is Wrong! Please Enter the locations for the Valid Flight Plan File!')
#                print('Example:\nG:\\00_BERKELEY\\01_RESEARCH\\03_Efficiency\\05_TFMS_Traj\\FlightPlans\\FlightPlans_IAH_BOS_2013\\FP_PT.csv')
#                VFPPath = raw_input()
#                self.VTrack = pd.read_csv(VFPPath)
            
        else:
            if db == True:
                client = pymongo.MongoClient()
                db = client.TFMS_TZ
                self.collection = db.FTRACK
                if Insert == True:
                    self.Track = self.UpToMongo() 
            else:
                # Get the data (flight track, inefficiency) from scratch
                self.Track = self.LoadData()
            
            df = self.Track.groupby('FID').head(1)
            ID_map = {}
            for idx, row in df.iterrows():
                ID_map[row.ID2] = row.FID
            
            self.Efficiency = self.GetEfficiency(self.Track.ID2.unique().tolist(), ID_map, self.year)
            self.VTrack, self.InvalidFid, self.WFFID = self.GetValidTrack(Cutoff, Tcut, Dcut, Vcut)            
    
    def GetEfficiency(self,Id, ID_map, year):
        client = pymongo.MongoClient()
        db = client.EnRoute
        if year == 2013:
            collection = db.ETMS2013_core2core
        elif year == 2014:
            collection = db.ETMS2014_core2core
        elif year == 2015:
            collection = db.ETMS2015_core2core
        else:
            raise ValueError('Efficiency Data year NOT FOUND')
        DOC = collection.aggregate(
               [ 
                 {'$project':{ 'Identifier': { '$concat': [ {"$substr":["$Flight_index",0,-1]}, "-", "$ACID" ]},
                               'DEPT_APRT':1, 'ARR_APRT':1, 'D40_A100_ACT_DIST':1, 'D40_A100_ACH_DIST':1,
                               'D40_A100_LF_ACT_DIST':1, 'D40_A100_GC_DIST':1 ,'_id':0}},
                 {'$match': { 'Identifier': {'$in':Id}, 'DEPT_APRT':self.Dep,'ARR_APRT':self.Arr}}
               ])
        i = 0
        Effi = {}
        for query in DOC:
            i += 1
            try:
                Effi[ID_map[query['Identifier']]] = [float(query['D40_A100_ACT_DIST']), float(query['D40_A100_ACH_DIST']), (float(query['D40_A100_ACT_DIST'])-float(query['D40_A100_ACH_DIST']))/float(query['D40_A100_ACH_DIST'])]
            except:
                pass
        return Effi    
    
    
    def LoadData(self):
        print('Start Loading Data...')
        ColIdx = np.append(np.arange(10),11)
        ColName = ['FID','FlightIdx','ACID','ACT_DATE','DEP','ARR',
                   'Elap_Time','Lat','Lon','Alt','GroundSpeed']
        Fname1 = 'ETMS_FlightTracks_'+ self.Dep + self.Arr + '_' + str(self.year) + '.csv'
        Fname2 = 'ETMS_FlightTracks_'+ self.Arr + self.Dep + '_' + str(self.year) + '.csv'
        try:
            OD_TRACK = pd.read_csv(os.getcwd() + '\TFMS\\' + Fname1,usecols=ColIdx,header=0, 
                                   names=ColName, parse_dates = [6])
        except IOError:
            OD_TRACK = pd.read_csv(os.getcwd() + '\TFMS\\' + Fname2,usecols=ColIdx,header=0, 
                                   names=ColName, parse_dates = [6])
            
        OD_TRACK = OD_TRACK[(OD_TRACK.DEP == self.Dep)&(OD_TRACK.ARR == self.Arr)].sort_values(by = ['FID','Elap_Time']).reset_index(drop = 1)

        DT = OD_TRACK.groupby('FID')['Elap_Time'].apply(lambda x: (x - x.shift(1)).dt.seconds)
        DT = DT.reset_index(drop = 1)
        
        DDist = OD_TRACK.groupby('FID').apply(lambda x: self.GCDist(x))   
        DDist = DDist.reset_index(drop = 1)
        
        OD_TRACK['ID2'] = OD_TRACK.FlightIdx.map(str) + '-' + OD_TRACK.ACID
        OD_TRACK['DT'] = DT
        OD_TRACK['Dist'] = DDist
        OD_TRACK['Speed'] = OD_TRACK.Dist/OD_TRACK.DT # nmi/sec
        OD_TRACK.Speed = OD_TRACK.Speed.replace([np.inf, -np.inf], np.nan)
                
        OD_TRACK[['DT','Dist','Speed']] = OD_TRACK[['DT','Dist','Speed']].fillna(0)
        OD_TRACK['CumDist'] = OD_TRACK.groupby('FID').Dist.cumsum()
        return OD_TRACK
   
    def GCDist(self,DF):
        start_lat = np.radians(DF.Lat)  
        start_lon = np.radians(DF.Lon)
        end_lat = np.radians(DF.Lat.shift(1))
        end_lon = np.radians(DF.Lon.shift(1))
        d_lat = end_lat - start_lat  
        d_lon = end_lon - start_lon  
        a = np.sin(d_lat/2)**2 + np.cos(start_lat) * np.cos(end_lat) * np.sin(d_lon/2)**2  
        c = 2 * np.arcsin(np.sqrt(a))  
        # Radius of earth is 3440 nm
        return 3440 * c
        
    def GetValidTrack(self, Cutoff, Tcut, Dcut, Vcut):
        print('-----------Start Cleaning-----------\n')
        Track1 = self.Track.groupby(['FID']).head(5)
        Track2 = self.Track.groupby(['FID']).tail(5)
        InValid_Track1 = Track1[~((Track1['Lat']> self.Ori_Lat - Cutoff) & (Track1['Lat'] < self.Ori_Lat + Cutoff) & 
                             (Track1['Lon']>self.Ori_Lon - Cutoff) & (Track1['Lon'] < self.Ori_Lon + Cutoff))].reset_index(drop = 1)
        InValid_Track2 = Track2[~((Track2['Lat']> self.Des_Lat - Cutoff) & (Track2['Lat'] < self.Des_Lat + Cutoff) & 
                             (Track2['Lon']> self.Des_Lon - Cutoff) & (Track2['Lon'] < self.Des_Lon + Cutoff))].reset_index(drop = 1)
#        InValid_Track3 = self.Track[self.Track.EnRoute_EFF.apply(np.isnan) == 1]
        
        InValid_Fid1 = InValid_Track1.FID.unique()
        InValid_Fid2 = InValid_Track2.FID.unique()
#        InValid_Fid3 = InValid_Track3.FID.unique()
#        InValid_Fid = functools.reduce(np.union1d,(InValid_Fid1,InValid_Fid2,InValid_Fid3))
        
        WFFID = self.DetectDiscont(Tcut, Dcut, Vcut)
        
        InValid_Fid = functools.reduce(np.union1d,(InValid_Fid1,InValid_Fid2,WFFID))
        
        VTRACK = self.Track[~((self.Track['FID'].isin(InValid_Fid)))]
        VTRACK = VTRACK[VTRACK.FID.isin(self.Efficiency.keys())].reset_index(drop = True)
        
        print('-----------Number of Invalid FID-----------')
        print(len(WFFID),len(InValid_Fid))
        print('-----------Number of Valid FID-----------')
        print(len(VTRACK.FID.unique()))
        return VTRACK, WFFID, InValid_Fid

    def DetectDiscont(self,Time_Cut, D_Cut, V_Cut):
        WFFID1 = self.Track[self.Track.DT > Time_Cut].FID.unique()
        WFFID2 = self.Track[self.Track.Dist > D_Cut].FID.unique()        
        WFFID3 = self.Track[self.Track.Speed > V_Cut].FID.unique()
        DistTrav = self.Track.groupby('FID').Dist.sum()
        WFFID4 = DistTrav.index[np.where(DistTrav < GCDistance(self.Ori_Lat,self.Ori_Lon,self.Des_Lat,self.Des_Lon))]
        WFFID = functools.reduce(np.union1d,(WFFID1,WFFID2,WFFID3,WFFID4))
        return WFFID

    def DetectHolding(self, Plot = False):
    	Holding = []
    	for gpidx, gp in self.VTrack.groupby('FID'):
    	    if LineString(gp[['Lon','Lat']].values).is_simple:
    	        pass
    	    else:
    	        Holding.append(gpidx)
    	if Plot:
    		self.PlotWithID(Holding)
    		return Holding
    	else:
    		return Holding

    def PlotWithID(self, FID_set = [], FlightIDX_ACID_set = [],xlb = -126,xrb = -65,ylb = 23.5,yub = 50, LW = 1, Animated = False, many_col = False):
        fig = plt.figure(figsize=(16,12))
        m = Basemap(llcrnrlon = xlb,llcrnrlat = ylb,urcrnrlon = xrb,urcrnrlat = yub,projection='merc')
        m.bluemarble()
        m.drawcoastlines(linewidth=0.5)
        m.drawcountries(linewidth=0.5)
        m.drawstates(linewidth=0.5)
        m.drawparallels(np.arange(10.,35.,5.))
        m.drawmeridians(np.arange(-120.,-80.,10.))

                          
        DrawSample = self.VTrack[(self.VTrack.FID.isin(FID_set) | (self.VTrack.FlightIdx.astype(str)+'-'+self.VTrack.ACID).isin(FlightIDX_ACID_set))][['FID','Lon','Lat']].set_index('FID')
        if many_col:
            C = plt.cm.Reds(np.linspace(0, 1, DrawSample.index.unique().shape[0]))
        else:
            C = ['r'] * DrawSample.index.unique().shape[0]
        i = -1
        for fid in DrawSample.index.unique():
            i += 1
            x,y = m(DrawSample.ix[fid,'Lon'].values, DrawSample.ix[fid,'Lat'].values)
            Trajectories, = plt.plot(x,y,'-', linewidth = LW, color = C[i], 
                             label = 'Real-time Trajectories: ' + self.Dep+'->'+self.Arr)
            if Animated:
                plt.pause(0.01)
        Ori_patch = np.array(Point([self.Ori_Lon, self.Ori_Lat]).buffer(0.5797).exterior.coords)
        Des_patch = np.array(Point([self.Des_Lon, self.Des_Lat]).buffer(1.4493).exterior.coords)
        Ori_x, Ori_y = m(Ori_patch[:,0],Ori_patch[:,1])
        Des_x, Des_y = m(Des_patch[:,0],Des_patch[:,1])
        OriPoly = PolygonPatch(Polygon(zip(Ori_x, Ori_y)), facecolor = 'y', alpha = 0.35, zorder = 15)
        DesPoly = PolygonPatch(Polygon(zip(Des_x, Des_y)), facecolor = 'y', alpha = 0.35, zorder = 15)
        plt.gca().add_patch(OriPoly)
        plt.gca().add_patch(DesPoly)
        return fig
        
        
    def Visualization(self,xlb = -126,xrb = -65,ylb = 23.5,yub = 50, PR = False, CDR = False, CoreAirport = False, Timeframe = 'All', QuickDraw = True):
        # Initialization 
        # Timeframe should be a list of interested month       
        fig = plt.figure(figsize=(16,12))
        m = Basemap(llcrnrlon = xlb,llcrnrlat = ylb,urcrnrlon = xrb,urcrnrlat = yub,projection='merc',resolution='l')
        m.bluemarble()
        # m.drawmapboundary(fill_color='aqua')
        # m.fillcontinents(lake_color='aqua')
        m.drawcoastlines(linewidth=0.5)
        m.drawcountries(linewidth=0.5)
        m.drawstates(linewidth=0.5)
        m.drawparallels(np.arange(10.,35.,5.))
        m.drawmeridians(np.arange(-120.,-80.,10.))
        GCRoute, = m.drawgreatcircle(self.Ori_Lon,self.Ori_Lat,self.Des_Lon,self.Des_Lat,
                          linewidth = 2.5,color='c',linestyle='--',zorder = 25, label = 'Great Circle Trajectory')
        
        if QuickDraw:
            # QuickDraw uses lat lon altogether, without considering the ordering of the trajectories.
            # Adjacent Line will be marked out as Reference Line.
            # QuickDraw is much faster than the others, especially when dataset is large.
            if Timeframe == 'All':
                longitude = self.VTrack.Lon.values
                latitude = self.VTrack.Lat.values
            else:
                longitude = np.array([])
                latitude = np.array([])
                for Month in Timeframe:
                    LeadingFID = int(str(self.year) + str(Month).zfill(2) + '00000000')
                    DrawSample = self.VTrack[(self.VTrack.FID >= LeadingFID) & (self.VTrack.FID < LeadingFID + 100000000)][['Lon','Lat']]
                    longitude = np.append(longitude,DrawSample.Lon)
                    latitude = np.append(latitude,DrawSample.Lat)
            x, y = m(longitude, latitude)
            
            if len(longitude) <= 3000000:
                Trajectories, = plt.plot(x,y,'-', linewidth = 0.3, color='r', 
                                     label = 'Real-time Trajectories: ' + self.Dep+'->'+self.Arr)
            else:
                Trajectories, = plt.plot(x,y,'.', markersize = 0.1, color='r', 
                                     label = 'Real-time Trajectories: ' + self.Dep+'->'+self.Arr)           
        else:
            if Timeframe == 'All':
                DrawSample = self.VTrack[['FID','Lat','Lon']].set_index('FID')
                for fid in DrawSample.index.unique():
                    x,y = m(DrawSample.ix[fid,'Lon'].values, DrawSample.ix[fid,'Lat'].values)
                    Trajectories, = plt.plot(x,y,'-', linewidth = 0.3, color='r', 
                                     label = 'Actual Trajectories: ' + self.Dep+'->'+self.Arr)

            else:
                longitude = np.array([])
                latitude = np.array([])
                for Month in Timeframe:
                    LeadingFID = int(str(self.year) + str(Month).zfill(2) + '00000000')
                    DrawSample = self.VTrack[(self.VTrack.FID >= LeadingFID) & (self.VTrack.FID < LeadingFID + 100000000)][['FID','Lat','Lon']].set_index('FID')
                    for fid in DrawSample.index.unique():
                        x,y = m(DrawSample.ix[fid,'Lon'].values, DrawSample.ix[fid,'Lat'].values)
                        Trajectories, = plt.plot(x,y,'-', linewidth = 0.3, color='r', 
                                     label = 'Real-time Trajectories: ' + self.Dep+'->'+self.Arr)
        
        LengendHandle = [Trajectories,GCRoute]
        
        if QuickDraw:
            x_ref, y_ref = m(np.array([self.Ori_Lon, self.Des_Lon]), np.array([self.Ori_Lat, self.Des_Lat]))
            RefLine, = plt.plot(x_ref,y_ref,
                          linewidth = 2,color='b',linestyle='-',zorder = 30, label = 'Reference Line')
            LengendHandle.append(RefLine)
            
        if PR == True:
            self.PR = PreferRoute()
            KEY1 = self.Dep + '_'+self.Arr            
            try:
                self.PrefRoute = self.PR.Pref_Route[KEY1]
            except:
                self.PrefRoute = {}
            
            if len(self.PrefRoute) == 0:
                print('No Preferred Routes Data Found')
            else:
                for i in range(len(self.PrefRoute)):
                    x_PR, y_PR = m(self.PrefRoute[str(i)][:,1],self.PrefRoute[str(i)][:,0])                
                    PRoute, = plt.plot(x_PR, y_PR,'-', zorder = 20,lw = 2, color = 'w', label = 'Preferred Routes')
                LengendHandle.append(PRoute)
                
        if CDR == True:
            self.PR = PreferRoute()
            KEY2 = 'K'+ self.Dep + '_K' + self.Arr
            try:
                self.CDRoute = self.PR.CD_Route[KEY2]
            except:
                self.CDRoute = {}
                
            if len(self.CDRoute) == 0:
                print('No Coded Departure Routes Data Found')
            else:
                for i in range(len(self.CDRoute)):
                    x_CDR, y_CDR = m(self.CDRoute[str(i)][:,1],self.CDRoute[str(i)][:,0])
                    CDRoute, = plt.plot(x_CDR, y_CDR,'-', zorder = 15,lw = 2, color = 'y', label = 'Coded Departure Routes')
                LengendHandle.append(CDRoute)
        
        if CoreAirport == True:
            AirportLocation = pd.read_csv('US_Core_Airport.csv', header = None, names = ['Airport','lat','lon'])
            lons = AirportLocation.lon.values
            lats = AirportLocation.lat.values
            x,y = m(lons, lats)
            CoreAir, = plt.plot(x, y, 'yo', markersize = 8, label = 'US Core 34 Airports')
            LengendHandle.append(CoreAir)
            
        LEGEND = plt.legend(handles=LengendHandle, loc = 0, fontsize = 20)            
        LEGEND.get_frame().set_facecolor('#C0C0C0')
        return fig
        
    def UpToMongo(self):
        Fname = 'ETMS_FlightTracks_'+self.Dep+self.Arr+'_'+str(self.year)+'.csv'
        i = 0
        FID = []
        Track = {}
        a = time.time()
        with open(os.getcwd() + '\TFMS\\' + Fname,'r') as csvfile:
            line = csv.reader(csvfile)
            next(line)
            for row in line:
                i += 1
                if i % 50000 == 0:
                    print(i)
                if [row[0],row[1]] not in FID:
                    FID.append([row[0],row[1]])
                    Track[row[0]] = {'FID':int(row[0]),'FIndex':row[1],'ACID':row[2],
                                     'ACT_Date':str(parse(row[3]))[:10],'DEP':row[4],'ARR':row[5],'TZ':[]}
                    Track[row[0]]['TZ'].append([parse(row[6]),float(row[7]),float(row[8]),float(row[9]),float(row[11])])
                else:
                    Track[row[0]]['TZ'].append([parse(row[6]),float(row[7]),float(row[8]),float(row[9]),float(row[11])])
        for KEY in Track:
            TZ = np.asarray(Track[KEY]['TZ'])[:,1:3].tolist()
            Track[KEY]['TZ_LS'] = {'type':'LineString','coordinates':TZ}
            self.collection.insert(Track[KEY])
        print(time.time() - a)
        return Track
        
    def SaveData(self):
        try:
            os.makedirs('TFMS_NEW')
        except:
            pass
        self.VTrack.to_csv(os.getcwd() + '\TFMS_NEW\\' + 'New_'+self.Dep+self.Arr+str(self.year)+'.csv',index = False)
        pickle.dump(self.Efficiency, open(os.getcwd() + '\TFMS_NEW\\' + 'Eff_'+self.Dep+self.Arr+str(self.year)+'.p','wb'))

