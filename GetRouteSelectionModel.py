# -*- coding: utf-8 -*-
"""
Created on Wed Sep 07 09:52:20 2016

@ Author: Liu, Yulin
@ Institute: UC Berkeley
"""
from __future__ import division
import pandas as pd
import numpy as np
import os
import geopandas as gpd
import shapely
import time
import collections
#import pylogit
"""
This is the file to concenate the NOAA weather data to trajectory data;
Argument the ETMS data by additional convective weather fields.
"""
class RouteSelectionModel:
    def __init__(self, DEP,ARR,DataDir = ['/ARTCC/artcc_cont.shp','/NOAA/ARTCC_Based_Weather_Sum_UTC.csv']):
        """
        DataDir should be a list of file directories
        """
        self.DEP = DEP
        self.ARR = ARR
        self.Dir = DataDir
        TrackFile = '/TFMS_NEW/New_' + self.DEP + self.ARR + '2013.csv'
        LabelFile = '/TFMS_NEW/Label_' + self.DEP + '_' + self.ARR + '_2013.csv'
        print('Loading Data...')
        self.ARTCC, self.All_Traj, self.ARTCC_TS, self.Med_ID_Dic, self.Nominal = self.LoadData(self.Dir[0],self.Dir[1],TrackFile,LabelFile)
        
        try:
            NominalDir = '/MNL/Nominal_' + DEP + '_' + ARR + '_2013.csv'
            self.Nominal_Dep = pd.read_csv(os.getcwd() + NominalDir, header = 0)
            print('Use Existing Nomianl Path Set')
        except:
            print('Start Constructing Nominal Path data...')        
            self.Nominal_Dep = self.NominalPath()
            self.Nominal_Dep.to_csv(os.getcwd() + NominalDir, index = False)

    def LoadData(self,ARTCC = '/ARTCC/artcc_cont.shp', Weather = '/NOAA/ARTCC_Based_Weather_Sum_UTC.csv', 
                 Track = '/TFMS_NEW/New_.csv', LabelTrack = '/TFMS_NEW/Label_.csv'):
        """
        Combine Nominal with ARTCC 
        """
        
        def GetARTCC(x, ARTCC):
            for idx, row in ARTCC.iterrows():
                if shapely.geometry.Point(x).intersects(row.geometry) == 1:
                    return row.ID
                else:
                    pass
        
        ARTCC = gpd.GeoDataFrame.from_file(os.getcwd()+ARTCC)
        ARTCC = ARTCC[['Name','Shape_Area','Shape_Leng','geometry']]
        ARTCC['ID'] = ARTCC.Name.apply(lambda x: x[-3:])

        TRACK = pd.read_csv(os.getcwd()+Track,header=0,usecols=[0,6,7,8,9,10,15], parse_dates = [1])
        All_Traj = pd.read_csv(os.getcwd() + LabelTrack,header=0,usecols=[0,2,6,7,8,11,16], parse_dates=[2])
        Med_ID_Dic = collections.OrderedDict(All_Traj[All_Traj.MedianID != -2][['FID','MedianID']].values)
        Nominal = TRACK[TRACK.FID.isin(Med_ID_Dic.keys())].reset_index(drop = 1)
        Nominal['ARTCC'] = Nominal[['Lon','Lat']].apply(lambda x: GetARTCC(x, ARTCC), axis = 1)
        # # ThunderStorm Data
        ARTCC_TS = pd.read_csv(os.getcwd() + Weather, header = 0)
        ARTCC_TS = ARTCC_TS.set_index(['ARTCC','UTCMonth','UTCDay','UTChour']).sort_index()
        return ARTCC, All_Traj, ARTCC_TS, Med_ID_Dic, Nominal
    # -----------------------------------------------------------------------------
    # Construct MNL Dataset
    # First to Construct Nominal Path dataframe to speed up
    # replace departure time and join together
    # Then append TS to Trajectory
    
    def NominalPath(self):
        i = 0
        ST = time.time()
        Nominal_copy = self.Nominal[['FID','Elap_Time','ARTCC']]
#        ChunkSize = Nominal_copy.shape[0]
#        temp = pd.DataFrame(np.zeros((self.All_Traj.shape[0] * ChunkSize,4)),columns = ['FID','Elap_Time','ARTCC','FID_Member'])
        
        for idx, row in self.All_Traj.iterrows():
            i += 1
            Nominal_copy.Elap_Time = Nominal_copy.groupby('FID').Elap_Time.apply(lambda x: x - (x.iloc[0] - row.Elap_Time)).reset_index(drop = True)
            Nominal_copy['FID_Member'] = row.values[0]
#            temp.iloc[(i-1) * ChunkSize: i* ChunkSize] = Nominal_copy
            if i == 1:
                temp = Nominal_copy.copy()
            else:
                temp = temp.append(Nominal_copy)
                
            if i % 1500 == 0:
                print(i, time.time() - ST)
        temp['UniqueKey'] = temp.Elap_Time.apply(lambda x: str(x)[:13]) + '_' + temp.ARTCC
        return temp
        
    def ConstructMNL_Data(self, SaveMNL = True):
        # # Query and Merge TS data
        
        WeatherField = self.ARTCC_TS.columns[range(2,18,1)]
        
        def IfTS(x,WeatherField, BeforeHour = 1, AfterHour = 1):
            # Per loop time should be around 0.025 seconds
            try:
                WeatherFeature = self.ARTCC_TS.ix[(x[14:],int(x[5:7]), int(x[8:10]),int(x[11:13]) - BeforeHour):(x[14:],int(x[5:7]), int(x[8:10]),int(x[11:13]) + AfterHour)][WeatherField].mean()
                NewData = pd.DataFrame(WeatherFeature).T
#                WeatherFeature = self.ARTCC_TS[(self.ARTCC_TS.ARTCC == x[14:]) & (self.ARTCC_TS.UTCDay == int(x[8:10])) &
#                                          (self.ARTCC_TS.UTCMonth == int(x[5:7])) & 
#                                          (self.ARTCC_TS.UTChour <= int(x[11:13]) + AfterHour) & 
#                                          (self.ARTCC_TS.UTChour >= int(x[11:13]) - BeforeHour)][WeatherField].sum()
            except:
                NewData = pd.DataFrame(np.array([np.arange(1)]*len(WeatherField)).T,columns = WeatherField)
            NewData['UniqueKey'] = x
            return NewData
        #-----------------------------------------
        print('Start Extracting Convective Weather ...')
        i = 0
        st = time.time()
        KEYS = map(str,self.Nominal_Dep['UniqueKey'].unique().tolist())        
        for key in KEYS:
            i += 1
            if i == 1:
                WeatherData = IfTS(key,WeatherField, 2, 2)
            else:
                WeatherData = WeatherData.append(IfTS(key,WeatherField, 2, 2))
            if i % 10000 == 0:
                print time.time() - st
        WeatherData = WeatherData.reset_index(drop = 1)
        
        temp1 = self.Nominal_Dep.merge(WeatherData, left_on='UniqueKey', right_on = 'UniqueKey', how = 'left')
        temp2 = temp1.groupby(['FID','FID_Member'])[WeatherField].mean().reset_index().sort_values(by = ['FID_Member','FID'])
        
        MNL_DATA = temp2.merge(self.All_Traj, left_on='FID_Member', right_on='FID', how = 'left')
        OutlierChoice = MNL_DATA.groupby('FID_Member').head(1).reset_index(drop = 1)
        OutlierChoice['FID_x'] = 99999999
        OutlierChoice[WeatherField] = 0.0
        MNL_DATA = MNL_DATA.append(OutlierChoice)
        MNL_DATA = MNL_DATA.sort_values(by = ['FID_Member','FID_x']).reset_index(drop = 1)
        MNL_DATA['CHOICE'] = 0
        MNL_DATA['Alt_id'] = 0
        
        OutlierGpIdx = self.Med_ID_Dic[max(self.Med_ID_Dic, key=self.Med_ID_Dic.get)]
        for gpidx, gp in MNL_DATA.groupby('FID_x'):
            try:
                MNL_DATA.ix[gp.index,'CHOICE'] = (gp.ClustID == self.Med_ID_Dic[gpidx]).map(int)
                MNL_DATA.ix[gp.index,'Alt_id'] = self.Med_ID_Dic[gpidx]
            except KeyError:
                MNL_DATA.ix[gp.index,'CHOICE'] = (gp.ClustID == -1).map(int)
                MNL_DATA.ix[gp.index,'Alt_id'] = OutlierGpIdx + 1

        # Argument the MNL dataset
        def GetSeason(x):
            if x >=12 or x<= 2:
                return 0 # Winter
            elif x <= 5:
                return 1 # Spring
            elif x <= 8:
                return 2 # Summer
            else:
                return 3 # Fall
                
        MNL_DATA['UTCMonth'] = MNL_DATA['Elap_Time'].apply(lambda x: x.month)
        MNL_DATA['UTCHour'] = MNL_DATA['Elap_Time'].apply(lambda x: x.hour)
        MNL_DATA['LocalMonth'] = MNL_DATA['Elap_Time'].apply(lambda x: x.tz_localize('UTC').astimezone('America/Chicago').month)
        MNL_DATA['LocalHour'] = MNL_DATA['Elap_Time'].apply(lambda x: x.tz_localize('UTC').astimezone('America/Chicago').hour)
        MNL_DATA['CA'] = MNL_DATA['ACID'].apply(lambda x: int(str(x)[:3]=='UAL'))
        MNL_DATA['Morning'] = MNL_DATA.LocalHour.apply(lambda x: int(x <= 12))
        MNL_DATA[['M1','M2','M3','M4','M5','M6','M7','M8','M9','M10','M11','M12']] = pd.get_dummies(MNL_DATA.LocalMonth).astype(np.int8)
        MNL_DATA['Season'] = MNL_DATA.LocalMonth.apply(lambda x: GetSeason(x))        
        MNL_DATA[['Winter','Spring','Summer','Fall']] = pd.get_dummies(MNL_DATA.Season).astype(np.int8)
        
        if SaveMNL == True:
            Fname = 'MNL_' + self.DEP + '_' + self.ARR+'_2013_Mean.csv'
            MNL_DATA.to_csv(os.getcwd() + '/MNL/'+Fname,index = False)
        return MNL_DATA
        
class LinearModel:
    def __init__(self, DEP,ARR, SaveLR = False):
        self.DEP = DEP
        self.ARR = ARR
        self.fname = 'MNL_' + self.DEP + '_' + self.ARR+'_2013_Mean.csv'
        self.LR_Data = self.GetLR_Data()
        if SaveLR == True:
            try:
                os.makedirs('LR')
            except:
                pass
            Fname = 'LR_' + self.DEP + '_' + self.ARR + '_2013.csv'
            self.LR_Data.to_csv(os.getcwd() + 'LR/'+Fname,index = False)
    
    def GetLR_Data(self):
        Data = pd.read_csv(os.getcwd() + '/MNL/' + self.fname, header = 0)
        
        wt = Data[Data.CHOICE == 1].groupby('FID_x').FID_Member.count()/Data.FID_Member.unique().shape[0]
        weight = lambda x: wt.ix[x]
        Data['wt'] = Data.groupby('FID_x').FID_x.transform(weight)
    
        ASC_ColumnIdx = range(2,18)
        SD_ColumnIdx = [23,22,29,30]
        
        Data[Data.columns[ASC_ColumnIdx]] = Data[Data.columns[ASC_ColumnIdx]].apply(lambda x: x * Data.wt)
        
        Keys = collections.OrderedDict()
        Keys['Efficiency'] = 'mean'
        
        for key in Data.columns[SD_ColumnIdx]:
            Keys[key] = 'mean'
        for key in Data.columns[ASC_ColumnIdx]:
            Keys[key] = 'sum'
        
        LR_Data = Data.groupby('FID_Member').agg(Keys).reset_index(drop = True)
        LR_Data['OD'] = self.fname[4:11]
        LR_Data['OD_Clust'] = self.fname[4:11] + '_' + LR_Data.ClustID.map(str)
        
        return LR_Data
    
##------------------------------------------------------------------------------
## # Multinomial Logit Model

#
#WeatherField = pd.Index([u'TS_sum', u'TS_mean', u'TSlevel_sum', u'TSlevel_mean', u'Hail_sum',
#       u'Hail_mean', u'Precipitation_sum', u'Precipitation_mean', u'Rain_sum',
#       u'Rain_mean', u'Shower_sum', u'Shower_mean', u'Ice_sum', u'Ice_mean',
#       u'Squall_sum', u'Squall_mean'],
#      dtype='object')
#MNL_DATA[WeatherField[range(0,len(WeatherField),2)]] = MNL_DATA[WeatherField[range(0,len(WeatherField),2)]] / 1000
#
#
## In[7]:
#
#BaseSpec = collections.OrderedDict()
#BaseName = collections.OrderedDict()
#
#BaseSpec['intercept'] = [0,1,2,3,4]
#BaseName['intercept'] = ['ASC_R0','ASC_R1','ASC_R2','ASC_R3','ASC_R4']
#
## for i in range(1,12):
##     exec "BaseSpec['M%d'] = [0]" %i
##     exec "BaseName['M%d'] = ['Month %d - fixed effects']" %(i,i)
#
#BaseSpec['Morning'] = [0,1,2,3,4]
#BaseName['Morning'] = ['MorningFlight_R0','MorningFlight_R1','MorningFlight_R2','MorningFlight_R3','MorningFlight_R4']
#
#BaseSpec['Spring'] = [0,1,2,3,4]
#BaseName['Spring'] = ['Spring_R0','Spring_R1','Spring_R2','Spring_R3','Spring_R4']
#
#BaseSpec['Summer'] = [0,1,2,3,4]
#BaseName['Summer'] = ['Summer_R0','Summer_R1','Summer_R2','Summer_R3','Summer_R4']
#
## Route 3 does not have winter trajectories
#BaseSpec['Fall'] = [0,1,2,4]
#BaseName['Fall'] = ['Fall_R0','Fall_R1','Fall_R2','Fall_R4']
#
## BaseSpec['CA'] = [0,1,2,3,4]
## BaseName['CA'] = ['United_R0','United_R1','United_R2','United_R3','United_R4']
#    
#BaseSpec['TS_mean'] = [[0,1,2,3,4]]
#BaseName['TS_mean'] = ['Thunder Storm']
#
## BaseSpec['Hail_mean'] = [[0,1,2,3,4]]
## BaseName['Hail_mean'] = ['Hail']
#
## BaseSpec['Precipitation_mean'] = [[0,1,2,3,4]]
## BaseName['Precipitation_mean'] = ['Precipitation']
#
#BaseSpec['Rain_mean'] = [[0,1,2,3,4]]
#BaseName['Rain_mean'] = ['Rain']
#
## BaseSpec['Shower_mean'] = [[0,1,2,3,4]]
## BaseName['Shower_mean'] = ['Shower']
#
## BaseSpec['Ice_sum'] = [[0,1,2,3,4]]
## BaseName['Ice_sum'] = ['Ice']
#
## BaseSpec['Squall_mean'] = [[0,1,2,3,4]]
## BaseName['Squall_mean'] = ['Squall']
#
#
## In[8]:
#
#MNL_Model = pylogit.create_choice_model(data = MNL_DATA, alt_id_col = 'Alt_id',obs_id_col = 'FID_Member',
#                                        choice_col = 'CHOICE',specification = BaseSpec, model_type = 'MNL', names = BaseName)
#
#
## In[9]:
#
#MNL_Model.fit_mle(np.zeros(len(BaseSpec) + 4*5-1), method='Newton-CG')
#Result = MNL_Model.get_statsmodels_summary()
#
#
## In[11]:
#
#Result.as_csv
#
#
## In[12]:
#
#Pred_dataset = MNL_DATA.copy()
#Pred_dataset['TS_mean'] = Pred_dataset['TS_mean'] * 0
#Predict_Result = MNL_DATA[['FID_x','Alt_id','CHOICE']].copy()
#Predict_Result['Probability'] = MNL_Model.predict(Pred_dataset)
#
#
## In[220]:
#
#Predict_Result.groupby(['FID_x','Alt_id']).Probability.mean()
#
#
## In[13]:
#
#Predict_Result.groupby(['FID_x','Alt_id']).Probability.mean()



