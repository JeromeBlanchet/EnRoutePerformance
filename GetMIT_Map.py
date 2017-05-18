# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 20:34:24 2017

@ Author: Liu, Yulin
@ Institute: UC Berkeley
"""
from __future__ import division
import pandas as pd
import geopandas as gpd
import numpy as np
import os
import time
import datetime
from shapely.geometry import LineString, Polygon, Point
from shapely.ops import cascaded_union
from sklearn.neighbors import KDTree
from collections import OrderedDict
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

def ProcessMITData():
    # Import MIT
    MIT_Enroute = pd.read_csv(os.getcwd()+'/TMI/MIT_Enroute_Merge.csv', sep = ',',parse_dates=[4,6])
    Airways = pd.read_csv(os.getcwd() + '/TMI/Geometry/Airways_CleanRelevant.csv')
    Navaids = pd.read_csv(os.getcwd() + '/TMI/Geometry/FixesNavaids_CleanRelevant.csv')
    TRACON = pd.read_csv(os.getcwd() + '/TMI/Geometry/TRACON.csv')
    
    # Convert TRACON into single polygon, with name unique

    def GetPolygon(x):
        BoundaryCoords = map(float,x.split())
        Poly = Polygon(zip(BoundaryCoords[1::2], BoundaryCoords[0::2]))
        if Poly.is_valid:
            return Poly
        else:
            mlon = 2 * sum(BoundaryCoords[1::2])/len(BoundaryCoords)
            mlat = 2 * sum(BoundaryCoords[0::2])/len(BoundaryCoords)
            def algo(x):
                return (math.atan2(x[0] - mlon, x[1] - mlat) + 2 * math.pi) % (2*math.pi)
            MeterPntList = zip(BoundaryCoords[1::2], BoundaryCoords[0::2])
            MeterPntList.sort(key=algo)
            return Polygon(MeterPntList)
    TRACON['geometry'] = TRACON.BOUNDARY.apply(lambda x: GetPolygon(x))
    TRACON_POLY = TRACON.groupby('TRACON_ID').geometry.apply(lambda x: cascaded_union(x)).reset_index()
    
    Facilities = np.unique(MIT_Enroute[['FRFAC','TOFACSINGLE']].values.reshape(1,-1))
    Fixes = MIT_Enroute[(MIT_Enroute.NAS_ELEM_TYPE == 'NAVAID')|(MIT_Enroute.NAS_ELEM_TYPE == 'FIX')].NAS_ELEM_SINGLE.unique()
    JetRoutes = MIT_Enroute[(MIT_Enroute.NAS_ELEM_TYPE == 'AIRWAY')].NAS_ELEM_SINGLE.unique()
#    Centers = MIT_Enroute[(MIT_Enroute.NAS_ELEM_TYPE == 'POLY')].NAS_ELEM_SINGLE.unique()
    Others = MIT_Enroute[pd.isnull(MIT_Enroute.NAS_ELEM_TYPE)].NAS_ELEM_SINGLE.unique()
    
    # Convert Jet Route into linestrings, with name unique
    Airways_Relevant = Airways[(Airways.NAME.isin(JetRoutes))|
                               (Airways.NAME.isin(Others))].sort_values(by = 
                                                                        ['NAME', 'DIRECTION', 'SEQ']).reset_index(drop = True)
    # Only keep the relavant (in terms of geometry data avalibility) jet routes/ nas element/ centers
    def ToLine(x):
        return LineString(zip(x.LON, x.LAT))
    
    Airways_Relevant_geo = Airways_Relevant.groupby(['NAME','DIRECTION'])[['LON','LAT']].apply(lambda x: ToLine(x)).reset_index()
    Airways_Relevant_geo.columns = ['NAME','DIRECTION','geometry']
    
    Navaids_Relevant = Navaids[(Navaids.NAME.isin(Fixes))|(Navaids.NAME.isin(Others))].reset_index(drop = True)[['NAME','LAT','LON']]
    TRACON_Relevant = TRACON_POLY[TRACON_POLY.TRACON_ID.isin(Facilities)].reset_index(drop = True)
    ARTCC_Relevant = gpd.GeoDataFrame.from_file(os.getcwd()+'/ARTCC/artcc_cont.shp')[['Name','geometry']]
    # ZAN (AK) and ZHN (HI) no data avaliable
    ARTCC_Relevant['Name'] = ARTCC_Relevant['Name'].apply(lambda x: x[-3:])
    TRACON_Relevant.columns = ['Name','geometry']
    Navaids_Relevant = Navaids[(Navaids.NAME.isin(Fixes))|(Navaids.NAME.isin(Others))].reset_index(drop = True)[['NAME','LAT','LON']]
    Navaids_Relevant['geometry'] = Navaids_Relevant.apply(lambda x: Point(x.LON, x.LAT), axis = 1)
    Navaids_Relevant = Navaids_Relevant[['NAME','geometry']]
    
    Facility_Relevant = ARTCC_Relevant.append(TRACON_Relevant).reset_index(drop = True)
    NAS_ELEM_Relevant = Navaids_Relevant.append(Airways_Relevant_geo).reset_index(drop = True)
    # Merge MIT with geometry files
    MIT_Enroute_geo = MIT_Enroute.merge(Facility_Relevant, \
                       left_on = 'FRFAC', right_on = 'Name', how = 'left').merge(\
                           Facility_Relevant, left_on = 'TOFACSINGLE', right_on = 'Name', how = 'left').merge(\
                                NAS_ELEM_Relevant, left_on = 'NAS_ELEM_SINGLE', right_on = 'NAME', how = 'left')
                                
    MIT_Enroute_geo.columns.values[-1] = 'NAS_ELEM_geo'
    MIT_Enroute_geo.columns.values[-4] = 'Pro_FAC_geo'
    MIT_Enroute_geo.columns.values[-6] = 'Req_FAC_geo'
    MIT_Enroute_geo = MIT_Enroute_geo.drop(['Name_x','Name_y','NAME'], axis = 1)
    MIT_Enroute_geo.NAS_ELEM_geo.fillna(MIT_Enroute_geo.Req_FAC_geo, inplace = True)
    MIT_Enroute_geo = MIT_Enroute_geo[~MIT_Enroute_geo.ALTITUDE.isin(['RAL','RALT'])].reset_index(drop = 1)
    
    # Convert altitude restrictions into numerical w/. sign
    def GetAltSign(x):
        if pd.isnull(x):
            return np.nan
        elif type(x) != str:
            return 0
        else:
            if '+' in x:
                return 1
            else:
                return -1
    def GetAltVal(x):
        if pd.isnull(x):
            return np.nan
        elif type(x) != str:
            return x
        else:
            if '+' in x:
                return int(x[:-1])
            else:
                return -int(x[:-1])        
    MIT_Enroute_geo['ALTITUDE_SIGN'] = MIT_Enroute_geo.ALTITUDE.apply(lambda x: GetAltSign(x))
    MIT_Enroute_geo['ALTITUDE_VALUE']= MIT_Enroute_geo.ALTITUDE.apply(lambda x: GetAltVal(x))
    return MIT_Enroute_geo

class MappingMIT:
    def __init__(self, Dep, Arr, Year, MIT_Enroute, Type = 'Nominal'):
        # load data
        # MIT_Enroute Should be a pandas dataframe, load pickle file into memory is good enough.
        self.Dep = Dep
        self.Arr = Arr
        self.Year = Year
        self.Type = Type
        VTrackPath = os.getcwd() + '/TFMS_NEW/New_' + Dep + Arr + str(Year) + '.csv'
        self.VTrack = pd.read_csv(VTrackPath, parse_dates=[6])
        self.LabelData = pd.read_csv(os.getcwd() + '/TFMS_NEW/Label_' + Dep+'_' + Arr+ '_' + str(Year) + '.csv', parse_dates=[6])
        self.CenterTraj = self.VTrack[self.VTrack.FID.isin(self.LabelData[self.LabelData.MedianID != -2].FID.values)].reset_index(drop = 1)
        self.CenterFlightID = self.CenterTraj.FID.unique()
        # MIT_Enroute_geo = pickle.load(open(os.getcwd() + '/TMI/MIT_WithGeometry_Enroute.p'))        
        self.MIT_Enroute = MIT_Enroute
        self.MIT_Enroute_VALUE = self.MIT_Enroute[['HEADID','MITVAL','DURATION_HRS_TOT']].drop_duplicates() 
        # Convert Geometry, Build trees
        
        self.Traj_Line, self.Traj_Tree = self.ConvertGeometry()
        self.Mapping_result = {}
        
    def ProcessMIT(self):
        return
    
    def ConvertGeometry(self):
        # Convert coordinates into linestrings
        # Build a tree
        Traj_Line = {}
        Traj_Tree = {} # 2d dictionary. {tree, DT np array}
        if self.Type == 'Nominal':
            for fid in self.CenterTraj.FID.unique():
                traj_coords = np.array(zip(self.CenterTraj[self.CenterTraj.FID == fid].Lon, 
                                           self.CenterTraj[self.CenterTraj.FID == fid].Lat))
                Traj_Line[fid] = LineString(traj_coords)
                Traj_Tree[fid] = [KDTree(traj_coords), self.CenterTraj[self.CenterTraj.FID == fid].DT.cumsum().values,\
                                 self.CenterTraj[self.CenterTraj.FID == fid].Alt.values*100]
        elif self.Type == 'Actual':
            for fid in self.VTrack.FID.unique():
                traj_coords = np.array(zip(self.VTrack[self.VTrack.FID == fid].Lon, 
                                           self.VTrack[self.VTrack.FID == fid].Lat))
                Traj_Line[fid] = LineString(traj_coords)
                Traj_Tree[fid] = [KDTree(traj_coords), self.VTrack[self.VTrack.FID == fid].DT.cumsum().values,\
                                 self.VTrack[self.VTrack.FID == fid].Alt.values*100]
        else:
            raise ValueError('Type can be either Nominal or Actual')
            
        
        return Traj_Line, Traj_Tree
    
    def MatchMit(self, departure, Trajectory, Traj_KDtree, MIT, parameters):
        # Trajectory should be a LineString
        # TrajID should be the FID of the matching Trajectory (center)
        # MIT should be a np array containing MITID, TOFAC, FRFAC, NAS_ELEM, NAS_ELEM_TYPE, Alt, MITVAL, DURATION, ...        
        # if MIT.NAS_ELEM_TYPE == 'AIRWAY':
        if MIT[13] == 'AIRWAY':
            BUFFER = parameters['AIRWAY'][0] ##
            k = 1
        elif MIT[13] == 'POLY':
            BUFFER = 0  ##
            k = 0
        else:
            BUFFER = parameters['NAS']  ##
            k = 0
        try:
            if k == 0:
                CrossPt_NAS = Trajectory.intersection(MIT[17].buffer(BUFFER)).coords[0][:2]
            else:
                if Trajectory.intersection(MIT[17].buffer(BUFFER)).length >= parameters['AIRWAY'][1]: ##
                    CrossPt_NAS = Trajectory.intersection(MIT[17].buffer(BUFFER)).coords[0][:2]
                else:
                    return 0
            
            CrossPt_Pro = Trajectory.intersection(MIT[15]).coords[0][:2]
            CrossPt_Req = Trajectory.intersection(MIT[14]).coords[0][:2]
        except:
            return 0
        
        CrossPt_Combine = np.array((CrossPt_Pro, CrossPt_Req, CrossPt_NAS))
        nearestidx = Traj_KDtree[0].query(CrossPt_Combine)[1]
        if nearestidx[0] > nearestidx[1]:
            return 0
        else:
            DeltaSec = Traj_KDtree[1][nearestidx[-1]][0]
            Altitude = Traj_KDtree[2][nearestidx[-1]][0]
            
            if np.isnan(MIT[19]):
                CrossTime = departure + datetime.timedelta(seconds = DeltaSec)
                if CrossTime >= MIT[4] and CrossTime <= MIT[6]:
                    return MIT[0] # MIT_ID
                #    return MIT[[0,11,5]] # MIT_ID, VALUE, DURATION
                else:
                    return 0
            else:
                if MIT[18] == 0:
                    if Altitude >= MIT[19] - 1000 and Altitude <= MIT[19] + 1000:
                        CrossTime = departure + datetime.timedelta(seconds = DeltaSec)
                        if CrossTime >= MIT[4] and CrossTime <= MIT[6]:
                            return MIT[0] # MIT_ID
#                             return MIT[[0,11,5]] # MIT_ID, VALUE, DURATION
                        else:
                            return 0
                    else:
                        return 0
                else:
                    if Altitude * MIT[18] >= MIT[19]:
                        CrossTime = departure + datetime.timedelta(seconds = DeltaSec)
                        if CrossTime >= MIT[4] and CrossTime <= MIT[6]:
                            return MIT[0] # MIT_ID
#                             return MIT[[0,11,5]] # MIT_ID, VALUE, DURATION
                        else:
                            return 0
                    else:
                        return 0
        
    def Main(self, parameters = {'AIRWAY':[0.25, 0.75], 'NAS': 0.5}):
        
        Airborne = self.CenterTraj.groupby('FID').DT.sum() # seconds
        st = time.time()
        for i in range(self.LabelData.shape[0]):
            if i % 1500 == 0:
                print(i, time.time() - st)
            
            departureTime = self.LabelData.loc[i, 'Elap_Time']
            FFID = self.LabelData.loc[i, 'FID']
            if self.Type == 'Actual':
                EndTime = self.LabelData.loc[i, 'Elap_Time'] + datetime.timedelta(seconds = self.LabelData.loc[i, 'DT'])
                ValidMIT = self.MIT_Enroute[(self.MIT_Enroute.RSTN_START < EndTime) & (self.MIT_Enroute.RSTN_STOP > departureTime)]
                if ValidMIT.shape[0] == 0:
                    self.Mapping_result[FFID] = np.array([0])
                else:
                    self.Mapping_result[FFID] = []
                    for idx, mit in enumerate(ValidMIT.values):                        
                        self.Mapping_result[FFID].append(self.MatchMit(departureTime, 
                                                self.Traj_Line[FFID], self.Traj_Tree[FFID], mit, parameters))
                    self.Mapping_result[FFID] = np.unique(np.array(self.Mapping_result[FFID]))
            
            if self.Type == 'Nominal':
                self.Mapping_result[FFID] = {}
                for fid in self.CenterFlightID:
                    EndTime = departureTime + datetime.timedelta(seconds = Airborne.loc[fid])
                    ValidMIT = self.MIT_Enroute[(self.MIT_Enroute.RSTN_START < EndTime) & (self.MIT_Enroute.RSTN_STOP > departureTime)]
                    if ValidMIT.shape[0] == 0:
                        self.Mapping_result[FFID][fid] = np.array([0])
                    else:
                        self.Mapping_result[FFID][fid] = []
                        for idx, mit in enumerate(ValidMIT.values):                            
                            self.Mapping_result[FFID][fid].append(self.MatchMit(departureTime, self.Traj_Line[fid], self.Traj_Tree[fid], mit, parameters))
                        self.Mapping_result[FFID][fid] = np.unique(np.array(self.Mapping_result[FFID][fid]))
        return self.Mapping_result
        
    def Count_Max_MIT(self):
        max_mit = 0
        k = 0
        for FFID in self.Mapping_result.keys():
            if self.Type == 'Nominal':
                for fid in self.Mapping_result[FFID].keys():
                    count_mit = np.count_nonzero(self.Mapping_result[FFID][fid])
                    if count_mit != 0:
                        k += 1
                    if count_mit > max_mit:
                        max_mit = count_mit
                    else:
                        pass
            else:
                count_mit = np.count_nonzero(self.Mapping_result[FFID])
                if count_mit != 0:
                    k += 1
                if count_mit > max_mit:
                    max_mit = count_mit
                else:
                    pass
        return max_mit, k
    
    def ConvertToDataFrame(self):
        
        # Only work for Nominal now...
        df = pd.DataFrame()
        for FFID in self.Mapping_result:
            d = pd.DataFrame(dict([(k,pd.Series(v)) for k,v in self.Mapping_result[FFID].iteritems()])).T
            d['parent_FID'] = FFID
            df = pd.concat([df, d])
        df = df.reset_index().set_index(['parent_FID','index']).stack().reset_index()
        df.columns = ['FFID','FID_Cluster', 'Seq','MIT_HEAD_ID']
        df = df.merge(self.MIT_Enroute_VALUE, left_on='MIT_HEAD_ID', right_on='HEADID', how='left')
        df['MIT_Str'] = df['MITVAL'] * df['DURATION_HRS_TOT']
        Oprs = OrderedDict()
        Oprs['HEADID']= np.count_nonzero
        Oprs['MITVAL']= [sum, np.mean, max]
        Oprs['DURATION_HRS_TOT']= [sum, np.mean, max]
        Oprs['MIT_Str']= [sum, np.mean, max]
        df = df.groupby(['FFID','FID_Cluster']).agg(Oprs).reset_index()
        df['MIT_Count'] = df['HEADID'] - 1
        df = df.fillna(0)
        df.columns = df.columns.droplevel()
        df = df.drop(['count_nonzero'], axis = 1)
        df.columns = ['FFID','FID_Cluster','MIT_VAL_sum','MIT_VAL_mean','MIT_VAL_max',\
                      'MIT_DUR_sum','MIT_DUR_mean','MIT_DUR_max','MIT_Str_sum',\
                      'MIT_Str_mean','MIT_Str_max', 'MIT_Count']
        df = df.sort_values(by=['FFID','FID_Cluster'])
        
#        MIT_df = pd.DataFrame({(i,j): Map_MIT[i][j]
#                                for i in Map_MIT.keys()
#                                    for j in Map_MIT[i].keys()}).T
#        MIT_df = MIT_df.reset_index()
#        MIT_df.columns = ['FFID','FID_Cluster', 'MIT_HEAD_ID']
#        MIT_df = MIT_df.merge(self.MIT_Enroute_VALUE,left_on='MIT_HEAD_ID', right_on='HEADID',how = 'left')
#        MIT_df = MIT_df.fillna(0)
#        MIT_df['MIT_Count'] = MIT_df.MIT_HEAD_ID.apply(lambda x: int(x>0))
#        MIT_df = MIT_df.drop(['HEADID'], axis = 1)
#        MIT_df = MIT_df.sort_values(by=['FFID','FID_Cluster'])
        return df
        
    def MergeWithMNL(self):
        # Only work for Nominal now...
        MIT_MAP_DF = self.ConvertToDataFrame()
        MNL_data_withWind = pd.read_csv(os.getcwd() + '/MNL/NEW_MNL_' + self.Dep + self.Arr + str(self.Year) +'.csv')
        MNL_data_withWind=MNL_data_withWind.merge(MIT_MAP_DF, left_on=['FID_Member','FID_x'], right_on = ['FFID','FID_Cluster'],how='left')
        MNL_data_withWind[['MIT_VAL_sum','MIT_VAL_mean','MIT_VAL_max',\
                      'MIT_DUR_sum','MIT_DUR_mean','MIT_DUR_max','MIT_Str_sum',\
                      'MIT_Str_mean','MIT_Str_max', 'MIT_Count']] = \
            MNL_data_withWind[['MIT_VAL_sum','MIT_VAL_mean','MIT_VAL_max',\
                      'MIT_DUR_sum','MIT_DUR_mean','MIT_DUR_max','MIT_Str_sum',\
                      'MIT_Str_mean','MIT_Str_max', 'MIT_Count']].fillna(0)
        MNL_data_withWind = MNL_data_withWind.drop(['FID_y','FFID','FID_Cluster'], axis = 1)
        return MNL_data_withWind
        
def MIT_MappingSummary(Dep, Arr, Year, Plot = False):
    try:
        MNL_data_withWind = pd.read_csv(os.getcwd() + '/MNL/Final_MNL_' + Dep + Arr + '_' + str(Year) +'.csv')
    except:
        print('Further Development')
    print('============%d trajectories in total============'%MNL_data_withWind.FID_Member.unique().shape[0])
    print('============%d nominal trajectories in total============'%(MNL_data_withWind.FID_x.unique().shape[0]-1))
    
    Oprs = OrderedDict([('TotalTraj', lambda x: len(x)), \
                        ('MIT_Instance', lambda x: np.count_nonzero(x)), \
                        ('MIT_Zero', lambda x: sum(x == 0)), \
                        ('MIT_One', lambda x: sum(x == 1)), \
                        ('MIT_Two', lambda x: sum(x == 2)), \
                        ('MIT_More', lambda x: sum(x >= 3))])
    SummaryStat = MNL_data_withWind.groupby('FID_x').agg({'MIT_Count':Oprs, 'CHOICE': {'ChoicePerc':lambda x: sum(x)/len(x)}}).reset_index()
    SummaryStat.columns = ['ClusterFID'] + SummaryStat.columns.droplevel()[1:].tolist()
    SummaryStat = SummaryStat.loc[SummaryStat.ClusterFID != 99999999,:]
    if Plot:
        VTRACK = pd.read_csv(os.getcwd()+'\TFMS_NEW\\'+'New_'+Dep+Arr+str(Year)+'.csv')
        MedianTrack = VTRACK[VTRACK.FID.isin(SummaryStat.ClusterFID.values)][['FID','Lon','Lat']]
        fig = plt.figure(figsize=(18,6))
        ax1 = fig.add_subplot(1,2,1)
        m = Basemap(llcrnrlon = -126,llcrnrlat = 23.5,urcrnrlon = -65,urcrnrlat = 50,projection='merc')
        m.bluemarble()
        m.drawcoastlines(linewidth=0.5)
        m.drawcountries(linewidth=0.5)
        m.drawstates(linewidth=0.5)
        m.drawparallels(np.arange(10.,35.,5.))
        m.drawmeridians(np.arange(-120.,-80.,10.))
        Col = ['r','g','y','w','m','c']
        XLegend = []
        i = -1
        for fid in SummaryStat.ClusterFID.values:
            i += 1
            x1,y1 = m(MedianTrack.loc[MedianTrack.FID == fid, 'Lon'].values, MedianTrack.loc[MedianTrack.FID == fid, 'Lat'].values)
            ax1.plot(x1,y1,'-',linewidth = 2, color = Col[i])
            Legd = Col[i] + ' | ' + str(SummaryStat.loc[SummaryStat.ClusterFID == fid, 'ChoicePerc'].values[0]*100)[:5] + '%'
            XLegend.append(Legd)
            
        ax1.set_title('Nominal Routes from %s to %s in %d'%(Dep, Arr, Year))
        
        plt.hold = True
        ax2 = fig.add_subplot(1,2,2)
        barcolors = plt.cm.Greens(np.linspace(0.25, 1, 4))
        ax2 = SummaryStat.plot(x = 'ClusterFID', y = ['MIT_One','MIT_Two','MIT_More'], kind = 'bar', stacked=True, \
                               ax = ax2, rot=0, color = barcolors)
        ax3 = ax2.twinx()
        ax2.set_ylabel('MIT Count')
        ax3.set_ylabel('MIT Percentage')
        ax3.set_ylim(0, SummaryStat.MIT_Instance.max()/SummaryStat.loc[1,'TotalTraj'])
        patches, labels = ax2.get_legend_handles_labels()
        ax2.legend(patches, labels, loc='best')
        ax2.set_xlabel('Cluster ID | Perc', fontsize=12)
        ax2.set_xticklabels(XLegend, rotation=0)
        ax2.set_title('MIT summary statistics', fontsize=12)
        
    else:
        pass

    return SummaryStat

