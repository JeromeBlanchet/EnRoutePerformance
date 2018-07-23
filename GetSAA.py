# -*- coding: utf-8 -*-
"""
Created on Wed Nov 8 2017

@ Author: Liu, Yulin
@ Institute: UC Berkeley
"""
from __future__ import division
import pandas as pd
import numpy as np
import os
import time
import datetime
from shapely.geometry import LineString, Polygon, Point
from sklearn.neighbors import KDTree
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import matplotlib.patches as plt_patch
from collections import OrderedDict
import re
import math

from dateutil.parser import parse
from datetime import datetime, timedelta

## converting areas
import pyproj
import shapely.ops as ops
from functools import partial

def getArea(polygon_geo):
    # return the area in km^2
    geom_area = ops.transform(partial(pyproj.transform,
                                      pyproj.Proj(init='EPSG:4326'),
                                      pyproj.Proj(proj='aea',
                                                  lat1=polygon_geo.bounds[1],
                                                  lat2=polygon_geo.bounds[3])),
                              polygon_geo)
    return geom_area.area * 1e-6


def extNFDC(saaGeoNfdcFile = os.getcwd() + '/TMI/SUA/geometry/nfdc.sua.CCU_20130110',
            saaGeoNfdcOutFile = os.getcwd() + '/TMI/SUA/geometry/write_nfdc.sua.CCU_20130110.txt'):
    i = 0
    with open(saaGeoNfdcFile, 'rb') as csvfile:
        line = csv.reader(csvfile)
        with open(saaGeoNfdcOutFile, 'wb') as wcsvfile:
            wline = csv.writer(wcsvfile)
            for row in line:
                i += 1
                if len(row) == 1:
                    coords_lat = re.findall('\d\d-\d\d-\d\d.\d\d\dN|\d\d-\d\d-\d\d.\d\d\dS', row[0])
                    coords_lon = re.findall('\d\d\d-\d\d-\d\d.\d\d\dW|\d\d\d-\d\d-\d\d.\d\d\dE', row[0])
                else:
                    k = 0
                    coords_lat = []
                    coords_lon = []
                    while (k < len(row)) and (len(coords_lat) < 1):
                        coords_lat = re.findall('\d\d-\d\d-\d\d.\d\d\dN|\d\d-\d\d-\d\d.\d\d\dS', row[k])
                        coords_lon = re.findall('\d\d\d-\d\d-\d\d.\d\d\dW|\d\d\d-\d\d-\d\d.\d\d\dE', row[k])
                        k += 1

                if len(coords_lat) != 0 and len(coords_lon) != 0:
                    lon = int(coords_lon[0][:3]) + int(coords_lon[0][4:6])/60 + float(coords_lon[0][7:13])/3600
                    lat = int(coords_lat[0][:2]) + int(coords_lat[0][3:5])/60 + float(coords_lat[0][6:12])/3600
                    if 'N' in coords_lat[0] and 'W' in coords_lon[0]:
                        wline.writerow([row[0][:39].rstrip()] + [row[0][39:64].rstrip()] + [-lon] + [lat])
                    elif 'N' in coords_lat[0] and 'E' in coords_lon[0]:
                        wline.writerow([row[0][:39].rstrip()] + [row[0][39:64].rstrip()] + [lon] + [lat])
                    elif 'S' in coords_lat[0] and 'E' in coords_lon[0]:
                        wline.writerow([row[0][:39].rstrip()] + [row[0][39:64].rstrip()] + [lon] + [-lat])
                    elif 'S' in coords_lat[0] and 'W' in coords_lon[0]:
                        wline.writerow([row[0][:39].rstrip()] + [row[0][39:64].rstrip()] + [-lon] + [-lat])
                    else:
                        raise ValueError("Error")

class extractFAASUAgeo:
    def __init__(self, saaGeoFile):
        # saaGeoFile = os.getcwd() + '/TMI/SUA/geometry/sua_definitions_56DySubscription_September_18__2014_-_November_13__2014.csv'
        self.saaGeoFile = saaGeoFile
        self.saaGeo = pd.read_csv(saaGeoFile,
                             usecols = [1, 2, 4, 9,18])

    def __GetPolygon(self, x):
        BoundaryCoords = map(float,re.split('; | ', x))
        Poly = Polygon(zip(BoundaryCoords[1::2], BoundaryCoords[0::2]))
    #     return Poly
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
    def __GetAirName(self, x, y, Type):
        if Type == 'M':
            return x.split(",")[0]
        elif Type == 'L':
            return y
        elif Type == 'R':
            return y[1:]
        elif Type == 'P':
            return y
        elif Type == 'W':
            return y[1:]
        else:
            return y

    def __GetType(self, x):
        if x == 'MOA':
            return 'M'
        elif x == 'AA':
            return 'L'
        elif x == 'ATCAA':
            return 'A'
        elif x == 'RA':
            return 'R'
        elif x == 'PA':
            return 'P'
        elif x == 'WA':
            return 'W'
        else:
            return 'O'
    
    def processSAAGeo(self):
        self.saaGeo['SUA_Name'] = ''
        self.saaGeo['Type'] = ''

        for idx, item in self.saaGeo.iterrows():
            item.Type = self.__GetType(item.sua_type)
            item.SUA_Name = self.__GetAirName(item.airspace_name, item.designator, item.Type)

        self.saaGeo = self.saaGeo.loc[self.saaGeo.operation == 'BASE'].drop_duplicates().reset_index(drop = 1)
        self.saaGeo['geometry_conv'] = self.saaGeo.geometry.apply(lambda x: self.__GetPolygon(x))
        

        saaATCAA = self.saaGeo[self.saaGeo.sua_type == 'MOA'].reset_index(drop = 1)
        saaATCAA['sua_type'] = 'ATCAA'
        saaATCAA['Type'] = 'A'
        saaATCAA['SUA_Name'] = saaATCAA['SUA_Name'].apply(lambda x: x[:-3] + 'ATCAA')
        self.saaGeo = self.saaGeo.append(saaATCAA).reset_index(drop = 1)

        return self.saaGeo[['SUA_Name', 'Type', 'geometry_conv']]

class extractNFDCSUAgeo:
    def __init__(self, saaGeoFile):
        # saaGeoFile = os.getcwd() + '/TMI/SUA/geometry/write_nfdc.sua.CCU_20130627.txt'
        self.saaGeoFile = saaGeoFile
        self.saaGeo = pd.read_csv(saaGeoFile, header=None, names=['sua_type', 'Name', 'Lon', 'Lat'])

    def __GetNFDCPolygon(self, NFDCdf):
        SUA_nfdc_geo = []
        for idx, gp in NFDCdf.groupby(['sua_type', 'Name']):
            gpLon = gp['Lon'].values.tolist()
            gpLat = gp['Lat'].values.tolist()
            Poly = Polygon(gp[['Lon', 'Lat']].values)
            if Poly.is_valid:
                pass
            else:
                mlon = sum(gpLon)/len(gpLon)
                mlat = sum(gpLat)/len(gpLat)
                def algo(x):
                    return (math.atan2(x[0] - mlon, x[1] - mlat) + 2 * math.pi) % (2*math.pi)
                MeterPntList = zip(gpLon, gpLat)
                MeterPntList.sort(key=algo)
                Poly = Polygon(MeterPntList)

            SUA_nfdc_geo.append(list(idx) + [Poly])
        SUA_nfdc_geo_pd = pd.DataFrame(SUA_nfdc_geo, columns=['sua_type', 'Name', 'geometry_conv'])
        return SUA_nfdc_geo_pd

    def __GetAirName(self, x, Type):
        if Type == 'SUA7MILITARY OPERATIONS AREA' and ('MOA' not in x):
            return x + ' MOA'
        elif Type == 'SUA7ALERT AREA':
            return 'A' + x
        elif Type == 'SUA7PROHIBITED AREA':
            return 'P' + x
        elif Type == 'SUA7RESTRICTED AREA':
            return x
        elif Type == 'SUA7WARNING AREA':
            return x
        else:
            return x

    def __GetType(self, x):
        if x == 'SUA7MILITARY OPERATIONS AREA':
            return 'M'
        elif x == 'SUA7ALERT AREA':
            return 'L'
        elif x == 'SUA7PROHIBITED AREA':
            return 'P'
        elif x == 'SUA7RESTRICTED AREA':
            return 'R'
        elif x == 'SUA7WARNING AREA':
            return 'W'
        else:
            return 'O'
    
    def processSAAGeo(self):
        self.saaGeo = self.__GetNFDCPolygon(self.saaGeo)

        self.saaGeo['Type'] = ''
        self.saaGeo['SUA_Name'] = ''

        for idx, item in self.saaGeo.iterrows():
            item.Type = self.__GetType(item.sua_type)
            item.SUA_Name = self.__GetAirName(item.Name, item.sua_type)

        saaATCAA = self.saaGeo[self.saaGeo.sua_type == 'SUA7MILITARY OPERATIONS AREA'].reset_index(drop = 1)
        saaATCAA['sua_type'] = 'ATCAA'
        saaATCAA['Type'] = 'A'
        saaATCAA['SUA_Name'] = saaATCAA['SUA_Name'].apply(lambda x: x[:-3] + 'ATCAA')
        self.saaGeo = self.saaGeo.append(saaATCAA).reset_index(drop = 1)

        return self.saaGeo[['SUA_Name', 'Type', 'geometry_conv']]

def combineSaaGeo(faaGeo, nfdcGeo, Plot = False):
    # In later version, should have the ability to merge multiple geometry data sources
    allGeo = nfdcGeo.append(faaGeo)
    allGeoUnique = allGeo.groupby(['SUA_Name', 'Type']).head(1).reset_index(drop = True)
    if Plot:
        plt.figure(figsize=(18,12))
        m = Basemap(llcrnrlon = -130, llcrnrlat = 22.5, urcrnrlon = -63, urcrnrlat = 50, projection='merc')
        m.drawmapboundary(fill_color='#8aeaff')
        m.fillcontinents(color='#c5c5c5', lake_color='#8aeaff')

        m.drawcoastlines(linewidth=0.5)
        m.drawcountries(linewidth=0.5)
        m.drawstates(linewidth=0.1)

        for idx, map_single in enumerate(allGeoUnique.values):
            if map_single[-1].type == 'Polygon':
                coords = np.array(map_single[-1].boundary.coords)
                x,y = m(coords[:,0], coords[:,1])
                xy = zip(x,y)
                if map_single[1] == 'M':
                    map_Poly = plt_patch.Polygon( xy, facecolor='blue', alpha=0.75, label = 'MOA', zorder = 2)
                elif map_single[1] == 'L':
                    map_Poly = plt_patch.Polygon( xy, facecolor='green', alpha=0.75, label = 'AA', zorder = 2)
                elif map_single[1] == 'R':
                    map_Poly = plt_patch.Polygon( xy, facecolor='yellow', alpha=0.75, label = 'RA', zorder = 2)
                elif map_single[1] == 'P':
                    map_Poly = plt_patch.Polygon( xy, facecolor='red', alpha=0.75, label = 'PA', zorder = 2)
                elif map_single[1] == 'W':
                    map_Poly = plt_patch.Polygon( xy, facecolor='white', alpha=0.75, label = 'WA', zorder = 2)
                elif map_single[1] == 'O':
                    map_Poly = plt_patch.Polygon( xy, facecolor='black', alpha=0.75, label = 'Others', zorder = 2)
                elif map_single[1] == 'A':
                    map_Poly = plt_patch.Polygon( xy, facecolor='magenta', alpha=0.75, label = 'ATCAA', zorder = 3)
                else:
                    map_Poly = plt_patch.Polygon( xy, facecolor='cyan', alpha=0.75, label = map_single[1], zorder = 1)
                plt.gca().add_patch(map_Poly)
                
            # elif map_single[-1].type == 'MultiPolygon':
            #     for single_poly in map_single[-1]:
            #         coords = np.array(single_poly.boundary.coords)
            #         x,y = m(coords[:,0], coords[:,1])
            #         xy = zip(x,y)
            #         if map_single[2] == 'MOA':
            #             map_Poly = plt_patch.Polygon( xy, facecolor='blue', alpha=0.75, label = 'MOA')
            #         elif map_single[2] == 'AA':
            #             map_Poly = plt_patch.Polygon( xy, facecolor='green', alpha=0.75, label = 'AA')
            #         elif map_single[2] == 'RA':
            #             map_Poly = plt_patch.Polygon( xy, facecolor='yellow', alpha=0.75, label = 'RA')
            #         elif map_single[2] == 'PA':
            #             map_Poly = plt_patch.Polygon( xy, facecolor='red', alpha=0.75, label = 'PA')
            #         elif map_single[2] == 'WA':
            #             map_Poly = plt_patch.Polygon( xy, facecolor='white', alpha=0.75, label = 'WA')
            #         elif map_single[2] == 'NSA':
            #             map_Poly = plt_patch.Polygon( xy, facecolor='black', alpha=0.75, label = 'NSA')
            #         else:
            #             map_Poly = plt_patch.Polygon( xy, facecolor='magenta', alpha=0.75, label = map_single[2])
            #         plt.gca().add_patch(map_Poly)
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), loc = 4)
        plt.show()
    return allGeoUnique
    

import tarfile
import gzip
import sys
def ParseSAAactivity(saaTarFile = os.getcwd() + "/TMI/SUA/sua2013.tar",
                     saaOutFile = os.getcwd() + '/TMI/SUA/SUA_combined_2013.csv'):
    tar = tarfile.open(saaTarFile)
    i = 0
    SUA_merge = []
    for member in tar.getmembers():
        if member.name.endswith('.csv.gz'):
            i += 1
            f = tar.extractfile(member)
            file_contents = gzip.GzipFile(fileobj = f).read().split("\n")
            if i == 1:
                pass
            else:
                file_contents = file_contents[1:]
            SUA_merge.extend(file_contents)
        if i % 5000 == 0:
            print("finish parsing %d" %i)

    # Unique SAA
    with open(saaOutFile, 'wb') as wfile:
        writer = csv.writer(wfile)
        writer.writerow(SUA_merge[0].split(","))
        for item in SUA_merge_unique:
            if len(item) == 0:
                pass
            else:
                row = item.split(",")[:-1]
                writer.writerow(row)

    ## Duplicates a lot. Deprecated
    # with open(saaOutFile, 'wb') as wfile:
    #     writer = csv.writer(wfile)
    #     for item in SUA_merge:
    #         row = item.split(",")[:-1]
    #         writer.writerow(row)


class preprocessSAAactivity:
    def __init__(self, saaOutFile, baseline = parse("01/01/2013 0:0:0")):
        self.baseline = baseline

        self.SUA_df = pd.read_csv(os.getcwd() + '/TMI/SUA/SUA_combined_2013_unique.csv', 
                     header = 0, names = ['Type', 'SUA_Name', 'StartTime', 'EndTime', "ARTCC", "State",
                                          'MinFL', 'MaxFL', 'Group'])
        self.SUA_df['SUA_Name'] = self.SUA_df['SUA_Name'].str.replace('=', '')
        self.SUA_df['SUA_Name'] = self.SUA_df['SUA_Name'].str.replace('"', '')

    def __ErrorExceptionParse(self, x):
        try:
            return parse(x)
        except:
            return parse("10/01/1949 0:0:0")

    def __cleanTime(self, SUA_df):

        SUA_df["StartTime"] = SUA_df["StartTime"].apply(lambda x: self.__ErrorExceptionParse(x))
        SUA_df["EndTime"] = SUA_df["EndTime"].apply(lambda x: self.__ErrorExceptionParse(x))
        SAA_df = SUA_df[(SUA_df['StartTime'] != parse("10/01/1949 0:0:0")) &
                        (SUA_df['EndTime'] != parse("10/01/1949 0:0:0"))].reset_index(drop = True)

        SAA_df.loc[SAA_df["StartTime"] <= parse("01/01/2013 0:0:0"), "StartTime"] = parse("01/01/2013 0:0:0")
        SAA_df.loc[SAA_df["EndTime"] <= parse("01/01/2013 0:0:0"), "EndTime"] = parse("01/01/2013 0:0:0")
        SAA_df.loc[SAA_df["EndTime"] >= parse("12/31/2013 23:59:59"), "EndTime"] = parse("12/31/2013 23:59:59")
        SAA_df.loc[SAA_df["StartTime"] >= parse("12/31/2013 23:59:59"), "StartTime"] = parse("12/31/2013 23:59:59")

        SAA_df['stDeltaRef'] = (SAA_df.StartTime - self.baseline).apply(lambda x: x.total_seconds())
        SAA_df['etDeltaRef'] = (SAA_df.EndTime - self.baseline).apply(lambda x: x.total_seconds())
        SAA_df['duration'] = SAA_df['etDeltaRef'] - SAA_df['stDeltaRef']
        SAA_df = SAA_df.loc[SAA_df.duration > 0].reset_index(drop = 1)
        return SAA_df

    def __mergeTime(self, SAA_df_clean):

        # Merge overlapped time intervals
        valid_records = []
        for gpidx, gp in SAA_df_clean.groupby("SUA_Name"):
            gp = gp.sort_values(by = "stDeltaRef")
            for idx, saa in enumerate(gp[['Type', 'SUA_Name', 'ARTCC', 
                                          'State', 'MinFL', 'MaxFL', 
                                          'Group', 'stDeltaRef', 'etDeltaRef']].values):
                
                if idx == 0:
                    tmpSaa = saa.copy()
                    curSt = saa[7]
                    curEt = saa[8]
                else:
                    if saa[7] > curEt:
                        valid_records.append(tmpSaa)
                        curSt = saa[7]
                        curEt = saa[8]
                        tmpSaa = saa.copy()
                    elif saa[8] <= curEt:
                        curSt = saa[7]
                        curEt = saa[8]
                        tmpSaa = saa.copy()
                        pass
                    else:
                        curEt = saa[8]
                        tmpSaa[8] = curEt
            valid_records.append(tmpSaa)
        valid_records = np.array(valid_records)
        SAA_clean_time = pd.DataFrame(valid_records, 
                                      columns=['Type', 'SUA_Name', 'ARTCC', 
                                               'State', 'MinFL', 'MaxFL', 
                                               'Group', 'stDeltaRef', 'etDeltaRef'])
        SAA_clean_time['duration'] = SAA_clean_time['etDeltaRef'] - SAA_clean_time['stDeltaRef']
        SAA_clean_time['StartTime'] = self.baseline + SAA_clean_time['stDeltaRef'].apply(lambda x: timedelta(seconds = x))
        SAA_clean_time['EndTime'] = self.baseline + SAA_clean_time['etDeltaRef'].apply(lambda x: timedelta(seconds = x))

        return SAA_clean_time

    def processSAA(self):
        SAA_df = self.SUA_df[(self.SUA_df['Group'] == 'SUA') & 
                                  (self.SUA_df['Type'] != 'AR')].reset_index(drop = True)
        print("clean time ...")
        _SAA_clean = self.__cleanTime(SAA_df)
        print("merge records ...")
        SAA_clean_time = self.__mergeTime(_SAA_clean)
        print("finish!")
        return SAA_clean_time

def mergeSAAwithGeo(allGeo, SAA):
    print('merge activity data with geometry data ...')
    SAA_merge = SAA.merge(allGeo[['SUA_Name', 'geometry_conv', 'Type']], 
                                     left_on=['SUA_Name', 'Type'], 
                                     right_on=['SUA_Name', 'Type'], 
                                     how = 'inner')
    print("%d out of %d records not matched"%(SAA.shape[0] - SAA_merge.shape[0], SAA.shape[0]))

    # missingMOA = SAA_merge[(SAA_merge.Type == 'M') & (pd.isnull(SAA_merge.geometry_conv))]['SUA_Name'].unique().tolist()
    # missingATCAA = SAA_merge[(SAA_merge.Type == 'A') & (pd.isnull(SAA_merge.geometry_conv))]['SUA_Name'].unique().tolist()
    print('parse altitude info ...')
    SAA_merge = SAA_merge.merge(SAA_merge.MinFL.apply(lambda x: pd.Series({'minRef': ("AGL" if '*' in x else 'MSL'), 
                                                           'minAlt':int(x[:-1] if '*' in x else int(x))})),
                left_index=True, right_index=True)
    SAA_merge = SAA_merge.merge(SAA_merge.MaxFL.apply(lambda x: pd.Series({'maxAlt':int(x[1:]) - 1 if '<' in x else int(x)})),
                    left_index=True, right_index=True)
    SAA_merge = SAA_merge.reset_index()
    SAA_merge = SAA_merge.rename(columns = {'index':'HEADID'})
    print('finish!')
    return SAA_merge


class MappingSAA:
    def __init__(self, Dep, Arr, Year, SAA, Type = 'Nominal'):
        # load data
        # SAA Should be a pandas dataframe, load pickle file into memory is good enough.
        self.Dep = Dep
        self.Arr = Arr
        self.Year = Year
        self.Type = Type
        self.VTrackPath = os.getcwd() + '/TFMS_NEW/New_' + Dep + Arr + str(Year) + '.csv'
        self.VTrack = pd.read_csv(self.VTrackPath, parse_dates=[6])
        self.LabelData = pd.read_csv(os.getcwd() + '/TFMS_NEW/Label_' + Dep+'_' + Arr+ '_' + str(Year) + '.csv', parse_dates=[6])
        self.CenterTraj = self.VTrack[self.VTrack.FID.isin(self.LabelData[self.LabelData.MedianID != -99].FID.values)].reset_index(drop = 1)
        self.CenterFlightID = self.CenterTraj.FID.unique()
        self.SAA = SAA[['HEADID', 'StartTime', 'EndTime', 'minAlt', 'maxAlt', 'geometry_conv', 'Type', 'SUA_Name']].copy()
        
        self.Traj_Line, self.Traj_Tree = self.ConvertGeometry()
        self.Mapping_result = {}
        
    def ProcessSAA(self):
        return

    def plotTrajwithSUA(self, allGeo, Type = 'actual', timeCover = 'all'):
        if timeCover == 'all':
            drawSAA = self.SAA
            if Type == 'actual':
                drawTraj = self.VTrack
            elif Type == 'nominal':
                drawTraj = self.CenterTraj
            elif Type == 'none':
                pass
            else:
                raise ValueError('Type can only be either nominal or actual')
        else:
            drawSAA = self.SAA.loc[(self.SAA.StartTime >= parse(timeCover[0])) & 
                                   (self.SAA.StartTime <= parse(timeCover[1]))]
            if Type == 'actual':
                drawTraj = self.VTrack.loc[self.VTrack.FID.isin(self.LabelData.loc[(self.LabelData.Elap_Time >= parse(timeCover[0])) & 
                                                             (self.LabelData.Elap_Time <= parse(timeCover[1]))].FID.values)]
            elif Type == 'nominal':
                drawTraj = self.CenterTraj
            elif Type == 'none':
                pass
            else:
                raise ValueError('Type can only be either nominal or actual')

        plt.figure(figsize=(18,12))
        m = Basemap(llcrnrlon = -130, llcrnrlat = 22.5, urcrnrlon = -63, urcrnrlat = 50, projection='merc')
        m.drawmapboundary(fill_color='#8aeaff')
        m.fillcontinents(color='#c5c5c5', lake_color='#8aeaff')

        m.drawcoastlines(linewidth=0.5)
        m.drawcountries(linewidth=0.5)
        m.drawstates(linewidth=0.1)
        for idx, map_single in enumerate(drawSAA[['SUA_Name', 'Type']].drop_duplicates().merge(allGeo, on = ['SUA_Name', 'Type']).values):
            if map_single[-1].type == 'Polygon':
                coords = np.array(map_single[-1].boundary.coords)
                x,y = m(coords[:,0], coords[:,1])
                xy = zip(x,y)
                if map_single[1] == 'M':
                    map_Poly = plt_patch.Polygon( xy, facecolor='blue', alpha=0.75, label = 'MOA', zorder = 2)
                elif map_single[1] == 'L':
                    map_Poly = plt_patch.Polygon( xy, facecolor='green', alpha=0.75, label = 'AA', zorder = 2)
                elif map_single[1] == 'R':
                    map_Poly = plt_patch.Polygon( xy, facecolor='yellow', alpha=0.75, label = 'RA', zorder = 2)
                elif map_single[1] == 'P':
                    map_Poly = plt_patch.Polygon( xy, facecolor='red', alpha=0.75, label = 'PA', zorder = 2)
                elif map_single[1] == 'W':
                    map_Poly = plt_patch.Polygon( xy, facecolor='white', alpha=0.75, label = 'WA', zorder = 2)
                elif map_single[1] == 'O':
                    map_Poly = plt_patch.Polygon( xy, facecolor='black', alpha=0.75, label = 'Others', zorder = 2)
                elif map_single[1] == 'A':
                    map_Poly = plt_patch.Polygon( xy, facecolor='m', alpha=0.75, label = 'ATCAA', zorder = 3)
                else:
                    map_Poly = plt_patch.Polygon( xy, facecolor='k', alpha=0.75, label = map_single[1], zorder = 1)
                plt.gca().add_patch(map_Poly)
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), loc = 4)

        if Type == 'none':
            pass
        else:
            for gpidx, gp in drawTraj.groupby('FID'):
                xlon, ylat = m(gp.Lon.values, gp.Lat.values)
                plt.plot(xlon, ylat, 'r-', zorder = 5)
        plt.show()

    def plotNominalwithSUA(self, allGeo, timeCover = 'all'):
        print('Deprecated, please use function plotTrajwithSUA with Type as nominal')
        if timeCover == 'all':
            drawSAA = self.SAA
        else:
            drawSAA = self.SAA.loc[(self.SAA.StartTime >= parse(timeCover[0])) & 
                                   (self.SAA.StartTime <= parse(timeCover[1]))]

        plt.figure(figsize=(18,12))
        m = Basemap(llcrnrlon = -130, llcrnrlat = 22.5, urcrnrlon = -63, urcrnrlat = 50, projection='merc')
        m.drawmapboundary(fill_color='#8aeaff')
        m.fillcontinents(color='#c5c5c5', lake_color='#8aeaff')

        m.drawcoastlines(linewidth=0.5)
        m.drawcountries(linewidth=0.5)
        m.drawstates(linewidth=0.1)
        for idx, map_single in enumerate(drawSAA[['SUA_Name', 'Type']].drop_duplicates().merge(allGeo, on = ['SUA_Name', 'Type']).values):
            if map_single[-1].type == 'Polygon':
                coords = np.array(map_single[-1].boundary.coords)
                x,y = m(coords[:,0], coords[:,1])
                xy = zip(x,y)
                if map_single[1] == 'M':
                    map_Poly = plt_patch.Polygon( xy, facecolor='blue', alpha=0.75, label = 'MOA', zorder = 2)
                elif map_single[1] == 'L':
                    map_Poly = plt_patch.Polygon( xy, facecolor='green', alpha=0.75, label = 'AA', zorder = 2)
                elif map_single[1] == 'R':
                    map_Poly = plt_patch.Polygon( xy, facecolor='yellow', alpha=0.75, label = 'RA', zorder = 2)
                elif map_single[1] == 'P':
                    map_Poly = plt_patch.Polygon( xy, facecolor='red', alpha=0.75, label = 'PA', zorder = 2)
                elif map_single[1] == 'W':
                    map_Poly = plt_patch.Polygon( xy, facecolor='white', alpha=0.75, label = 'WA', zorder = 2)
                elif map_single[1] == 'O':
                    map_Poly = plt_patch.Polygon( xy, facecolor='black', alpha=0.75, label = 'Others', zorder = 2)
                elif map_single[1] == 'A':
                    map_Poly = plt_patch.Polygon( xy, facecolor='m', alpha=0.75, label = 'ATCAA', zorder = 3)
                else:
                    map_Poly = plt_patch.Polygon( xy, facecolor='k', alpha=0.75, label = map_single[1], zorder = 1)
                plt.gca().add_patch(map_Poly)
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), loc = 4)

        for gpidx, gp in self.CenterTraj.groupby('FID'):
            xlon, ylat = m(gp.Lon.values, gp.Lat.values)
            plt.plot(xlon, ylat, 'g--', zorder = 5)
        plt.show()
    
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
                Traj_Tree[fid] = [KDTree(traj_coords), 
                                  self.CenterTraj[self.CenterTraj.FID == fid].DT.cumsum().values,
                                  self.CenterTraj[self.CenterTraj.FID == fid].Alt.values*100]
        elif self.Type == 'Actual':
            for fid in self.VTrack.FID.unique():
                traj_coords = np.array(zip(self.VTrack[self.VTrack.FID == fid].Lon, 
                                           self.VTrack[self.VTrack.FID == fid].Lat))
                Traj_Line[fid] = LineString(traj_coords)
                Traj_Tree[fid] = [KDTree(traj_coords), 
                                  self.VTrack[self.VTrack.FID == fid].DT.cumsum().values,
                                  self.VTrack[self.VTrack.FID == fid].Alt.values*100]
        else:
            raise ValueError('Type can be either Nominal or Actual')

        return Traj_Line, Traj_Tree
    
    def MatchSAA(self, departure, Trajectory, Traj_KDtree, saa, how = 'single', **kwargs):
        # Trajectory should be a LineString
        # TrajID should be the FID of the matching Trajectory (center)
        # SAA should be a np array containing 0:HEADID, 1:StartTime, 2:EndTime, 3:minAlt, 4:maxAlt, 5:geometry, 6:Type
        if saa[-1] == 'M':
            saa_type = 1
        elif saa[-1] == 'A':
            saa_type = 2
        elif saa[-1] == 'R':
            saa_type = 3
        elif saa[-1] == 'W':
            saa_type = 4
        elif saa[-1] == 'L':
            saa_type = 5
        elif saa[-1] == 'P':
            saa_type = 6
        elif saa[-1] == 'O':
            saa_type = 7
        # intersection returns the line segment within the polygon
        if how == 'single':
            interLine = Trajectory.intersection(saa[5])

            try:
                EntryPt = interLine.coords[0][:2] # entry point
                ExitPt = interLine.coords[-1][:2] # exit point
            except NotImplementedError:
                interLine = list(interLine)
                try:
                    EntryPt = interLine[0].coords[0][:2] # entry point
                    ExitPt = interLine[-1].coords[-1][:2] # exit point
                except IndexError:
                    return [-1, -1, -1]
            
            CrossPt_Combine = np.array((EntryPt, ExitPt))
            # find two closest points (index) for entry/ exit point
            nearestidx = Traj_KDtree[0].query(CrossPt_Combine, k = 2)[1]
            
            # Use the average of the time of the two closest points for entry/exit point as the altitude
            Entry_Altitude = np.mean(Traj_KDtree[2][nearestidx[0]])
            Exit_Altitude = np.mean(Traj_KDtree[2][nearestidx[1]])
            # if saa_type == 2:
            #     print(Entry_Altitude, saa[(3,4),], CrossPt_Combine)
            if (Entry_Altitude >= saa[3]*100 and Entry_Altitude <= saa[4]*100) or (Exit_Altitude >= saa[3]*100 and Exit_Altitude <= saa[4]*100):

                # print("altitude is ok %d"%saa[0])
                # Use the average of the time of the two closest points for entry point as the crossing time
                Entry_DeltaSec = np.mean(Traj_KDtree[1][nearestidx[0]])
                Exit_DeltaSec = np.mean(Traj_KDtree[1][nearestidx[1]])

                EntryTime = departure + timedelta(seconds = Entry_DeltaSec)
                ExitTime = departure + timedelta(seconds = Exit_DeltaSec)

                if EntryTime < saa[2] and ExitTime > saa[1]:
                    # Entry time should be earlier than ALERT STOP TIME and exit time should be later than ALERT START TIME
                    TraverseTime = abs(Exit_DeltaSec - Entry_DeltaSec)
                    if TraverseTime <= 0:
                        return [-1, -1, -1]
                    else:
                        return [saa[0], TraverseTime, saa_type]
                else:
                    return [-1, -1, -1]
            else:
                return [-1, -1, -1]

        elif how == 'buffer':
            # try
            radius = kwargs['rad']
            trajectoryRod = Trajectory.buffer(radius)
            # rad could be around 0.5 degree
            interPoly = trajectoryRod.intersection(saa[5])
            # centroid point
            # area

            centroidPT = interPoly.centroid.coords

            if len(centroidPT) == 0:
                return [-1, -1, -1]
            else:
                interArea = getArea(interPoly)
                centroidPT = np.array(centroidPT[0]).reshape(1, -1)
                nearestidx = Traj_KDtree[0].query(centroidPT, k = 2)[1]
                # print(nearestidx)
                centeroidAltitude = np.mean(Traj_KDtree[2][nearestidx])

                if (centeroidAltitude >= saa[3]*100 and centeroidAltitude <= saa[4]*100):

                    # print("altitude is ok %d"%saa[0])
                    # Use the average of the time of the two closest points for entry point as the crossing time
                    centroidDeltaSec = np.mean(Traj_KDtree[1][nearestidx])
                    centroidTime = departure + timedelta(seconds = centroidDeltaSec)

                    if centroidTime <= saa[2] and centroidTime >= saa[1]:
                        # Entry time should be earlier than ALERT STOP TIME and exit time should be later than ALERT START TIME
                        return [saa[0], interArea, saa_type]
                    else:
                        return [-1, -1, -1]
                else:
                    return [-1, -1, -1]

        else:
            raise ValueError('how can only be single or buffer')
        
    def Main(self, how = 'single', **kwargs):
        
        Airborne = self.CenterTraj.groupby('FID').DT.sum() # seconds
        st = time.time()
        for i in range(self.LabelData.shape[0]):
            if i % 20 == 0:
                print(i, time.time() - st)
            
            departureTime = self.LabelData.loc[i, 'Elap_Time']
            FFID = self.LabelData.loc[i, 'FID']
            if self.Type == 'Actual':
                EndTime = self.LabelData.loc[i, 'Elap_Time'] + timedelta(seconds = self.LabelData.loc[i, 'DT'])
                ValidSAA = self.SAA[(self.SAA.StartTime < EndTime) & (self.SAA.EndTime > departureTime)][['HEADID', 'StartTime', 'EndTime', 'minAlt', 'maxAlt', 'geometry_conv', 'Type']]
                if ValidSAA.shape[0] == 0:
                    self.Mapping_result[FFID] = np.array([[-1, -1, -1]])
                else:
                    self.Mapping_result[FFID] = []
                    for idx, saa in enumerate(ValidSAA.values):                        
                        self.Mapping_result[FFID].append(self.MatchSAA(departureTime, 
                                                                       self.Traj_Line[FFID], 
                                                                       self.Traj_Tree[FFID], 
                                                                       saa,
                                                                       how,
                                                                       **kwargs))
                    self.Mapping_result[FFID] = np.unique(np.array(self.Mapping_result[FFID]), axis=0)
            
            elif self.Type == 'Nominal':
                self.Mapping_result[FFID] = {}
                for fid in self.CenterFlightID:
                    EndTime = departureTime + timedelta(seconds = Airborne.loc[fid])
                    ValidSAA = self.SAA[(self.SAA.StartTime < EndTime) & (self.SAA.EndTime > departureTime)][['HEADID', 'StartTime', 'EndTime', 'minAlt', 'maxAlt', 'geometry_conv', 'Type']]
                    
                    if ValidSAA.shape[0] == 0:
                        self.Mapping_result[FFID][fid] = np.array([[-1, -1, -1]])
                    else:
                        # print(ValidSAA.shape[0])
                        self.Mapping_result[FFID][fid] = []
                        for idx, saa in enumerate(ValidSAA.values):                            
                            self.Mapping_result[FFID][fid].append(self.MatchSAA(departureTime, 
                                                                                self.Traj_Line[fid], 
                                                                                self.Traj_Tree[fid], 
                                                                                saa,
                                                                                how,
                                                                                **kwargs))
                        self.Mapping_result[FFID][fid] = np.unique(np.array(self.Mapping_result[FFID][fid]), axis=0)
        return self.Mapping_result
    
    def Count_Max_SAA(self):
        max_saa = 0
        k = 0 # nonzero SAA traj.
        for FFID in self.Mapping_result.keys():
            if self.Type == 'Nominal':
                for fid in self.Mapping_result[FFID].keys():
                    count_saa = np.count_nonzero(self.Mapping_result[FFID][fid][:,0] != -1)
                    if count_saa != 0:
                        k += 1
                    if count_saa > max_saa:
                        max_saa = count_saa
                    else:
                        pass
            else:
                count_saa = np.count_nonzero(self.Mapping_result[FFID][:,0] != -1)
                if count_saa != 0:
                    k += 1
                if count_saa > max_saa:
                    max_saa = count_saa
                else:
                    pass
        return max_saa, k
    
    def ConvertToDataFrame(self):
        # 
        if self.Type == 'Nominal':
            df = []
            for FFID in self.Mapping_result:
                for fid in self.Mapping_result[FFID]:
                    summary_stat = [0] * 18
                    summary_stat[0] = FFID
                    summary_stat[1] = fid
                    
                    for k in range(7):
                        summary_stat[2+k] = np.count_nonzero(self.Mapping_result[FFID][fid][:,2] == 1+k)    
                    summary_stat[9] = np.count_nonzero(self.Mapping_result[FFID][fid][:,2] != -1)
                    
                    for k in range(7):
                        summary_stat[10 + k] = np.mean(self.Mapping_result[FFID][fid][:,1][np.where(self.Mapping_result[FFID][fid][:,2]==1+k)])
                    summary_stat[17] = np.mean(self.Mapping_result[FFID][fid][:,1][np.where(self.Mapping_result[FFID][fid][:,2]!=-1)])
                    
                    df.append(summary_stat)
            df = np.array(df)
            df[np.isnan(df)] = 0
            data_df = pd.DataFrame(data = df, 
                                   columns=['FFID',
                                            'FID_Cluster',
                                            
                                            'NumM',
                                            'NumA',
                                            'NumR',
                                            'NumW',
                                            'NumL',
                                            'NumP',
                                            'NumO',
                                            'NumTot',
                                            
                                            'AvgTimeM',
                                            'AvgTimeA',
                                            'AvgTimeR',
                                            'AvgTimeW',
                                            'AvgTimeL',
                                            'AvgTimeP',
                                            'AvgTimeO',
                                            'AvgTimeTot'
                                            ])
            
            data_df = data_df.astype(dtype = {'FFID':'int64',
                                            'FID_Cluster':'int64',
                                            
                                            'NumM':'int32',
                                            'NumA':'int32',
                                            'NumR':'int32',
                                            'NumW':'int32',
                                            'NumL':'int32',
                                            'NumP':'int32',
                                            'NumO':'int32',
                                            'NumTot':'int32',
                                            
                                            'AvgTimeM':'float',
                                            'AvgTimeA':'float',
                                            'AvgTimeR':'float',
                                            'AvgTimeW':'float',
                                            'AvgTimeL':'float',
                                            'AvgTimeP':'float',
                                            'AvgTimeO':'float',
                                            'AvgTimeTot':'float'})
        else:
            raise NotImplementedError("Not Implemented")
        return data_df
    
    def MergeWithMNL(self, SAVE = False, data_match_saa = None, MNL_data = None):
        # Only work for Nominal now...
        if data_match_saa is None:
            SUA_MAP_DF = self.ConvertToDataFrame()
        else:
            SUA_MAP_DF = data_match_saa.copy()

        if MNL_data is None:
            MNL_data_withWind = pd.read_csv(os.getcwd() + '/MNL/Final_MNL_' + self.Dep + self.Arr + '_' + str(self.Year) +'.csv')
        else:
            MNL_data_withWind = MNL_data.copy()
        MNL_data_withWind = MNL_data_withWind.merge(SUA_MAP_DF, left_on=['FID_Member','FID_x'], right_on = ['FFID','FID_Cluster'],how='left')

        MNL_data_withWind = MNL_data_withWind.drop(['FFID','FID_Cluster'], axis = 1)
        if SAVE:
            MNL_data_withWind.to_csv(os.getcwd() + '/MNL/SAA_MA_WIND_WX_Final_MNL_' + self.Dep + self.Arr + '_' + str(self.Year) +'.csv', index = False)
        return MNL_data_withWind

