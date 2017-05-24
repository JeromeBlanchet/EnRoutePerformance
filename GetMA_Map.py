# -*- coding: utf-8 -*-
"""
Created on Fri May 19 15:31:24 2017

@ Author: Liu, Yulin
@ Institute: UC Berkeley
"""

from __future__ import division
import numpy as np
import pandas as pd
import geopandas as gpd
import csv
import matplotlib.pyplot as plt
import os
from shapely.geometry import LineString, Polygon, Point
from shapely.ops import cascaded_union
import pickle
import math
import matplotlib.pyplot as plt
import datetime
from mpl_toolkits.basemap import Basemap
from dateutil.parser import parse
import matplotlib.patches as plt_patch
from sklearn.neighbors import KDTree
import time
import argparse

def ProcessMAPData(SAVE = False):
	TRACON = pd.read_csv(os.getcwd() + '/TMI/Geometry/TRACON.csv')
	SECTOR = pd.read_csv(os.getcwd() + '/TMI/Geometry/SECTOR.csv', usecols=[0,1,2,3,7,8,9,10,11,12,13,14])
	SECTOR_MAP = pd.read_csv(os.getcwd() + '/TMI/MAP_Value/SECTOR_MAP.csv')
	SECTOR_MAP['SECTOR_NAME'] = SECTOR_MAP['CENTER'] + SECTOR_MAP['SECTOR'].map('{:02}'.format)

	# Convert TRACON/SECTOR into single polygon, with name unique
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

	# SECTOR
	SECTOR['SECTOR_NAME'] = SECTOR['CENTER'] + SECTOR['SECTOR'].map("{:02}".format)
	SECTOR['geometry'] = SECTOR.BOUNDARY.apply(lambda x: GetPolygon(x))
	SECTOR_POLY = SECTOR.groupby('SECTOR_NAME').geometry.apply(lambda x: cascaded_union(x)).reset_index()
	# TRACON
	TRACON['geometry'] = TRACON.BOUNDARY.apply(lambda x: GetPolygon(x))
	TRACON_POLY = TRACON.groupby('TRACON_ID').geometry.apply(lambda x: cascaded_union(x)).reset_index()
	# ARTCC
	ARTCC_POLY = gpd.GeoDataFrame.from_file(os.getcwd()+'/ARTCC/artcc_cont.shp')[['Name','geometry']]
	ARTCC_POLY['Name'] = ARTCC_POLY['Name'].apply(lambda x: str(x[-3:]))
	# Rename
	SECTOR_POLY.columns = ['NAME', 'geometry']
	TRACON_POLY.columns = ['NAME', 'geometry']
	ARTCC_POLY.columns = ['NAME', 'geometry']
	# Paste
	facility_poly = SECTOR_POLY.append(ARTCC_POLY).append(TRACON_POLY)

	# Merge geometry file with MAP data

	MAP_data = pd.read_csv(os.getcwd() + '/TMI/MAP_VALUE/MA_DATA_2013-2015.csv', usecols = [0,1,2,3,4,5,6], parse_dates=[2,6])
	MAP_data['start_year'] = MAP_data.ALERTTIME.apply(lambda x: x.year)
	MAP_data['end_year'] = MAP_data.STOPTIME.apply(lambda x: x.year)
	MAP_data_2013 = MAP_data[(MAP_data.start_year == 2013) & (MAP_data.end_year == 2013)].reset_index(drop = 1)
	print(MAP_data_2013.shape)

	MAP_2013_GEO = MAP_data_2013.merge(facility_poly, left_on='SECTOR', right_on='NAME', how = 'inner').reset_index(drop = 1)
	if SAVE:
		pickle.dump(MAP_2013_GEO, open(os.getcwd() + '/TMI/MAP_VALUE/MA2013_GEO.p', 'wb'), protocol = 2)

def VisualizeMAP(MAP, alert_start = '01/01/2013 00:00:00', alert_end = '01/01/2013 12:00:30'):
    visual_map = MAP[(MAP.ALERTTIME >= parse(alert_start)) & (MAP.ALERTTIME <= parse(alert_end))][['ALERT_TYPE', 'geometry', 'SECTOR','ENTRYID']]
    print('Total number of MAP: %d' %visual_map.shape[0])
    m = Basemap(llcrnrlon = -128, llcrnrlat = 22.5, urcrnrlon = -63, urcrnrlat = 50, projection='merc')
    m.bluemarble()
    m.drawcoastlines(linewidth=0.5)
    m.drawcountries(linewidth=0.5)
    m.drawstates(linewidth=0.2)
    m.drawparallels(np.arange(10.,35.,5.))
    m.drawmeridians(np.arange(-120.,-80.,10.))
    
    for idx, map_single in enumerate(visual_map.values):
        if map_single[1].type == 'Polygon':
            coords = np.array(map_single[1].boundary.coords)
            x,y = m(coords[:,0], coords[:,1])
            xy = zip(x,y)
            if map_single[0] == 'RED':
                map_Poly = plt_patch.Polygon( xy, facecolor='red', alpha=0.9)
            else:
                map_Poly = plt_patch.Polygon( xy, facecolor='yellow', alpha=0.9)
            plt.gca().add_patch(map_Poly)
        elif map_single[1].type == 'MultiPolygon':
            for single_poly in map_single[1]:
                coords = np.array(single_poly.boundary.coords)
                x,y = m(coords[:,0], coords[:,1])
                xy = zip(x,y)
                if map_single[0] == 'RED':
                    map_Poly = plt_patch.Polygon( xy, facecolor='red', alpha=0.9)
                else:
                    map_Poly = plt_patch.Polygon( xy, facecolor='yellow', alpha=0.9)
                plt.gca().add_patch(map_Poly)
    return visual_map

class MappingMAP:
    def __init__(self, Dep, Arr, Year, MA_Geo, Type = 'Nominal'):
        # load data
        self.Dep = Dep
        self.Arr = Arr
        self.Year = Year
        self.Type = Type
        VTrackPath = os.getcwd() + '/TFMS_NEW/New_' + Dep + Arr + str(Year) + '.csv'
        VTrack = pd.read_csv(VTrackPath, parse_dates=[6])
        self.LabelData = pd.read_csv(os.getcwd() + '/TFMS_NEW/Label_' + Dep+'_' + Arr+ '_' + str(Year) + '.csv', parse_dates=[6])
        self.CenterTraj = VTrack[VTrack.FID.isin(self.LabelData[self.LabelData.MedianID != -2].FID.values)].reset_index(drop = 1)
        self.CenterFlightID = self.CenterTraj.FID.unique()
        self.MAP = MA_Geo[['ENTRYID', 'ALERTTIME', 'STOPTIME', 'ALERT_TYPE', 'geometry']] 
        # notice the order of the column is different from the original MAP_2013_GEO data
        
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
        for fid in self.CenterTraj.FID.unique():
            traj_coords = np.array(zip(self.CenterTraj[self.CenterTraj.FID == fid].Lon, self.CenterTraj[self.CenterTraj.FID == fid].Lat))
            Traj_Line[fid] = LineString(traj_coords)
            Traj_Tree[fid] = [KDTree(traj_coords), self.CenterTraj[self.CenterTraj.FID == fid].DT.cumsum().values]
        return Traj_Line, Traj_Tree
    
    def MatchMAP(self, departure, Trajectory, Traj_KDtree, MA):
        # Trajectory should be a LineString
        # TrajID should be the FID of the matching Trajectory (center)
        # MA should be a np array containing ENTRYID, ALERTTIME, STOPTIME, ALERT_TYPE, geometry
        if MA[3] == 'RED':
            alert = 2
        else:
            alert = 1
        try:
            # intersection returns the line segment within the polygon
            EntryPt = Trajectory.intersection(MA[4]).coords[0][:2] # entry point
            ExitPt = Trajectory.intersection(MA[4]).coords[-1][:2] # exit point
        except:
            return (0,0,0)
        
        CrossPt_Combine = np.array((EntryPt, ExitPt))
        # find two closest points (index) for entry/ exit point
        nearestidx = Traj_KDtree[0].query(CrossPt_Combine, k = 2)[1]
        # Use the average of the time of the two closest points for entry point as the crossing time
        Entry_DeltaSec = np.mean(Traj_KDtree[1][nearestidx[0]])
        Exit_DeltaSec = np.mean(Traj_KDtree[1][nearestidx[1]])
        
        EntryTime = departure + datetime.timedelta(seconds = Entry_DeltaSec)
        ExitTime = departure + datetime.timedelta(seconds = Exit_DeltaSec)
        
        if EntryTime < MA[2] and ExitTime > MA[1]:
            # Entry time should be earlier than ALERT STOP TIME and exit time should be later than ALERT START TIME
            TraverseTime = abs(Exit_DeltaSec - Entry_DeltaSec)
            return (MA[0], TraverseTime, alert)
        else:
            return (0,0,0)
        
    def Main(self):
        
        Airborne = self.CenterTraj.groupby('FID').DT.sum() # seconds
        st = time.time()
        for i in range(self.LabelData.shape[0]):
            if i % 500 == 0:
                print(i, time.time() - st)
    
            departureTime = self.LabelData.loc[i, 'Elap_Time']
            FFID = self.LabelData.loc[i, 'FID']
            self.Mapping_result[FFID] = {}
            
            for fid in self.CenterFlightID:
                EndTime = departureTime + datetime.timedelta(seconds = Airborne.loc[fid])
                ValidMAP = self.MAP[(self.MAP.ALERTTIME < EndTime) & (self.MAP.STOPTIME > departureTime)]
                if ValidMAP.shape[0] == 0:
                    self.Mapping_result[FFID][fid] = np.array([[0,0,0]])
                else:
                    self.Mapping_result[FFID][fid] = []
                    for idx, MA in enumerate(ValidMAP.values):
                        self.Mapping_result[FFID][fid].append(self.MatchMAP(departureTime, self.Traj_Line[fid], self.Traj_Tree[fid], MA))
                    self.Mapping_result[FFID][fid] = np.array(list(set(self.Mapping_result[FFID][fid])))
        return self.Mapping_result
    
    def Count_Max_MA(self):
        max_ma = 0
        k = 0 # nonzero MA traj.
        for FFID in self.Mapping_result.keys():
            if self.Type == 'Nominal':
                for fid in self.Mapping_result[FFID].keys():
                    count_ma = np.count_nonzero(self.Mapping_result[FFID][fid][:,0])
                    if count_ma != 0:
                        k += 1
                    if count_ma > max_ma:
                        max_ma = count_ma
                    else:
                        pass
            else:
                count_ma = np.count_nonzero(self.Mapping_result[FFID][:,0])
                if count_ma != 0:
                    k += 1
                if count_ma > max_ma:
                    max_ma = count_ma
                else:
                    pass
        return max_ma, k
    
    def ConvertToDataFrame(self):
        if self.Type == 'Nominal':
            df = []
            for FFID in self.Mapping_result:
                for fid in self.Mapping_result[FFID]:
                    summary_stat = [0,0,0,0,0,0]
                    summary_stat[0] = FFID
                    summary_stat[1] = fid
                    summary_stat[2] = np.count_nonzero(self.Mapping_result[FFID][fid][:,2] == 2)
                    summary_stat[3] = np.count_nonzero(self.Mapping_result[FFID][fid][:,2] == 1)
                    summary_stat[4] = np.mean(self.Mapping_result[FFID][fid][:,1][np.where(self.Mapping_result[FFID][fid][:,2]==2)])
                    summary_stat[5] = np.mean(self.Mapping_result[FFID][fid][:,1][np.where(self.Mapping_result[FFID][fid][:,2]==1)])
                    df.append(summary_stat)
            df = np.array(df)
            df[np.isnan(df)] = 0

            data_df = pd.DataFrame(data = df, columns=['FFID','FID_Cluster','NumRed','NumYellow','AvgTimeRed','AvgTimeYellow'])

            data_df.astype(dtype = {'FFID':'int64','FID_Cluster':'int64','NumRed':'int','NumYellow':'int','AvgTimeRed':'float','AvgTimeYellow':'float'})
        else:
            print('Actual type is under development')
        return data_df
    
    def MergeWithMNL(self, SAVE = False):
        # Only work for Nominal now...
        MA_MAP_DF = self.ConvertToDataFrame()
        MNL_data_withWind = pd.read_csv(os.getcwd() + '/MNL/Final_MNL_' + self.Dep + self.Arr + '_' + str(self.Year) +'.csv')
        MNL_data_withWind = MNL_data_withWind.merge(MA_MAP_DF, left_on=['FID_Member','FID_x'], right_on = ['FFID','FID_Cluster'],how='left')
        MNL_data_withWind[['NumRed','NumYellow','AvgTimeRed','AvgTimeYellow']] = \
            MNL_data_withWind[['NumRed','NumYellow','AvgTimeRed','AvgTimeYellow']].fillna(0)
        MNL_data_withWind = MNL_data_withWind.drop(['FFID','FID_Cluster'], axis = 1)
        if SAVE:
            MNL_data_withWind.to_csv(os.getcwd() + '/MNL/MA_Final_MNL_' + self.Dep + self.Arr + '_' + str(self.Year) +'.csv', index = False)
        return MNL_data_withWind

### MAIN ###
def main():

    parser = argparse.ArgumentParser(description='Match Monitor Alert with Nominal Route')
    subparsers = parser.add_subparsers()

    common_args = [
        (['-DEP', '--Dep'], {'help':'Depature airport', 'type': str, 'default': 'IAH'}),
        (['-ARR', '--Arr'], {'help':'Arrival airport', 'type': str, 'default': 'BOS'}),
        (['-Year', '--year'], {'help':'year', 'type':int, 'default':2013}),
        (['-MA_Loc', '--MA_Loc'], {'help':'file location of MA', 'type':str, 'default':'/TMI/MAP_VALUE/MA2013_GEO.p'}),
        (['-match_type', '--TYPE'], {'help':'match type', 'type':str, 'default':'Nominal'}),
        (['-save', '--SAVE_MNL'], {'help':'Dump MNL data to file?', 'type':bool, 'default':True})
    ]

    parser_train = subparsers.add_parser('matching')
    parser_train.set_defaults(which='matching')
    for arg in common_args:
        parser_train.add_argument(*arg[0], **arg[1])

    parser_preprocess = subparsers.add_parser('preprocessing')
    parser_preprocess.set_defaults(which='preprocessing')
    parser_preprocess.add_argument('-Save_MA', '--SAVE_MA', help='Dump MA data to file?', type=bool, default=True)

    args = parser.parse_args()
    if args.which == 'preprocessing':
        ProcessMAPData(args.SAVE_MA)
        print('Preprocessing finished')
    elif args.which == 'matching':
    	print('Matching Monitor Alert for flights from %s to %s' % (args.Dep, args.Arr))
    	MA_Geo = pickle.load(open(os.getcwd() + args.MA_Loc, 'rb'))
    	MA_MATCH = MappingMAP(args.Dep, args.Arr, args.year, MA_Geo, args.TYPE)
    	match_result = MA_MATCH.Main()
    	print(MA_MATCH.Count_Max_MA())
    	MNL_data = MA_MATCH.MergeWithMNL(args.SAVE_MNL)

if __name__ == '__main__':
    main()