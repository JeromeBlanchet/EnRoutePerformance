# -*- coding: utf-8 -*-
"""
@ Author: Liu, Yulin
@ Institute: UC Berkeley
"""
from __future__ import division
import time
import pandas as pd
import os
import numpy as np
from dateutil.parser import parse
import re

def GetSeason(x):
    if x >=12 or x<= 2:
        return 0 # Winter
    elif x <= 5:
        return 1 # Spring
    elif x <= 8:
        return 2 # Summer
    else:
        return 3 # Fall

class GetMNLdataset:
    def __init__(self, DEP, ARR, Year, TimeZone):
    	"""
    	TimeZone:
    	America/New_York
    	America/Chicago
    	America/Los_Angeles
    	America/Denver
    	"""
    	print('------------------Constructing generic MNL dataset-------------------')
    	print('Options for TimeZone:')
    	print('America/New_York; America/Chicago; America/Los_Angeles; America/Denver')
        self.DEP = DEP
        self.ARR = ARR
        self.Year = Year
        self.TimeZone = TimeZone
        self.LabelData, self.CenterTraj, self.MemberFID = self.LoadTraj()

    def LoadTraj(self):
        print('---------------- Load Trajectories ----------------')
        VTrackPath = os.getcwd() + '/TFMS_NEW/New_' + self.DEP + self.ARR + str(self.Year) + '.csv'
        VTrack = pd.read_csv(VTrackPath, parse_dates=[6])
        LabelData = pd.read_csv(os.getcwd() + '/TFMS_NEW/Label_' + self.DEP+'_' + self.ARR+ '_' + str(self.Year) + '.csv', parse_dates=[6])
        CenterTraj = VTrack[VTrack.FID.isin(LabelData[LabelData.MedianID != -2].FID.values)].reset_index(drop = 1)
        MemberFID = VTrack.FID.unique()
        print('Finished')
        return LabelData, CenterTraj, MemberFID

    def ConstructingMNL(self, Save = False):
        MergingIdx = np.hstack((np.repeat(self.CenterTraj.groupby('FID').head(1).FID.values.reshape(1,-1), self.MemberFID.shape[0], axis = 0).reshape(-1,1), 
                                np.repeat(self.MemberFID.reshape(-1,1), self.CenterTraj.FID.unique().shape[0], axis = 0)))

        MNL_DATA = pd.DataFrame(MergingIdx, columns=['FID_x', 'FID'])
        MNL_DATA = MNL_DATA.merge(self.LabelData[['FID','ACID','Elap_Time','DT','Efficiency','ClustID']])
        MNL_DATA = MNL_DATA.merge(self.LabelData.loc[self.LabelData.MedianID != -2,['FID', 'MedianID']].drop_duplicates(), 
                       left_on='FID_x', right_on='FID', how = 'left')
        MNL_DATA.columns.values[1] = 'FID_Member'
        MNL_DATA.columns.values[-1] = 'Alt_id'
        MNL_DATA['CHOICE'] = (MNL_DATA.ClustID == MNL_DATA.Alt_id).map(int)

        MNL_DATA['UTCMonth'] = MNL_DATA['Elap_Time'].apply(lambda x: x.month)
        MNL_DATA['UTCHour'] = MNL_DATA['Elap_Time'].apply(lambda x: x.hour)
        MNL_DATA['LocalMonth'] = MNL_DATA['Elap_Time'].apply(lambda x: x.tz_localize('UTC').astimezone(self.TimeZone).month)
        MNL_DATA['LocalHour'] = MNL_DATA['Elap_Time'].apply(lambda x: x.tz_localize('UTC').astimezone(self.TimeZone).hour)
        MNL_DATA['CARRIER'] = MNL_DATA['ACID'].apply(lambda x: "".join(re.findall("[a-zA-Z]+", x)))
        MNL_DATA['Morning'] = MNL_DATA.LocalHour.apply(lambda x: int(x <= 12))
        MNL_DATA[['M1','M2','M3','M4','M5','M6','M7','M8','M9','M10','M11','M12']] = pd.get_dummies(MNL_DATA.LocalMonth).astype(np.int8)
        MNL_DATA['Season'] = MNL_DATA.LocalMonth.apply(lambda x: GetSeason(x))        
        MNL_DATA[['Winter','Spring','Summer','Fall']] = pd.get_dummies(MNL_DATA.Season).astype(np.int8)
        del MNL_DATA['FID_y']

        OutlierChoice = MNL_DATA.groupby('FID_Member').head(1).reset_index(drop = 1)
        OutlierChoice.loc[:,'FID_x'] = 99999999
        OutlierChoice.loc[:,'CHOICE'] = 0
        OutlierChoice.loc[:,'Alt_id'] = MNL_DATA.Alt_id.max() + 1
        MNL_DATA = MNL_DATA.append(OutlierChoice)
        MNL_DATA.loc[(MNL_DATA.ClustID == -1) & (MNL_DATA.FID_x == 99999999), 'CHOICE'] = 1
        MNL_DATA = MNL_DATA.sort_values(by = ['FID_Member','FID_x']).reset_index(drop = 1)

        if Save:
        	MNL_DATA.to_csv(os.getcwd() + '/MNL/MNL_Generic' + self.DEP + '_' + self.ARR + '_' + str(self.Year) + '.csv', index = False)

        return MNL_DATA