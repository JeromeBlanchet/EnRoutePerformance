# -*- coding: utf-8 -*-
"""
@author: Yulin
"""

from __future__ import division
import time
import pandas as pd
import os
import collections
import numpy as np
try:
    import cPickle as pickle
except:
    import pickle
from scipy.spatial import cKDTree
from sklearn.neighbors import KDTree
# KDTree from sklearn has far more features than that from scipy. cKDTree from scipy is faster, though.
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from dateutil.parser import parse


# Global param setting
global BaseTime 
global A
global B
global ESQ

BaseTime = parse('2013-01-01 00:00:00')
A = 6378.137
B = 6356.7523142
ESQ = 6.69437999014 * 0.001


# some usefule functions
def lookup(s):
    """
    This is an extremely fast approach to datetime parsing.
    For large data, the same dates are often repeated. Rather than
    re-parse these, we store all unique dates, parse them, and
    use a lookup to convert all dates.
    """
    dates = {date:pd.to_datetime(date) for date in s.unique()}
    return s.map(dates)

def geodetic2ecef(lon, lat, alt=0):
    """Convert geodetic coordinates to ECEF."""
    lat = np.radians(lat)
    lon = np.radians(lon)
    xi = np.sqrt(1 - ESQ * np.sin(lat))
    x = (A / xi + alt) * np.cos(lat) * np.cos(lon)
    y = (A / xi + alt) * np.cos(lat) * np.sin(lon)
    z = (A / xi * (1 - ESQ) + alt) * np.sin(lat)
    return x, y, z

def euclidean_distance(distance):
    """Return the approximate Euclidean distance corresponding to the
    given great circle distance (in km).
    """
    return 2 * A * np.sin(distance / (2 * B))

def great_circle_distance(euc_distance):
    """Return the approximate great circle distance corresponding to the
    given Euclidean distance (in km).
    """
    return 2 * B * np.arcsin(euc_distance / (2 * A))

def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a)) 
    r = 6371 # Radius of earth in kilometers. Use 3956 for miles
    return c * r


# ## Process weather data:
# 1. parse UTC time
# 2. construct delta time w.r.t. a base time
# 3. remove duplicated time (groupby and mean)
# 4. save valid data
def construct_wx(index_df, valid_weather_data, wx = 'TS_mean'):
    WX = index_df.merge(valid_weather_data[['WBAN', 'DT', wx]], on = ['WBAN','DT'], how = 'left')
    WX = WX.fillna(0)
    return WX[wx].values.reshape(index_df.WBAN.unique().shape[0], index_df.DT.unique().shape[0]).T

def ProcessWeatherData(Wx_stationwise_dir = os.getcwd() + '/NOAA/Station_Based_Weather_Sum_UTC_2013.csv', 
                       Wx_station_dir = os.getcwd() + '/NOAA/WeatherStation.csv', 
                       UTCyear = 2013,
                       Save = True):
    print('---------------- Preprocessing Wx file ----------------')
    print('---------------- Load convective weather files (Stationwise) ----------------')
    weather_data = pd.read_csv(Wx_stationwise_dir)
    weather_station = pd.read_csv(Wx_station_dir)

    weather_data['UTCtime'] = lookup(weather_data['UTCYear'].map(str) + '-' +  weather_data['UTCMonth'].map(str) + '-' + weather_data['UTCDay'].map(str) + ' ' + weather_data['UTChour'].map(str) + ':00:00')
    weather_data['DT'] = (weather_data.UTCtime - BaseTime).astype('timedelta64[s]')/3600.
    weather_data_valid = weather_data.groupby(['WBAN', 'DT'])['TS_mean','TSlevel_mean','Hail_mean',
                                         'Precipitation_mean','Rain_mean','Shower_mean',
                                         'Ice_mean','Squall_mean'].mean().reset_index()

    # Make Wx structured (prepare for fast query)
    valid_station = pd.DataFrame(data = {'WBAN': weather_data_valid.WBAN.unique()}).merge(weather_station[['WBAN','LON','LAT']], how = 'inner')
    valid_station = valid_station.sort_values(by = 'WBAN').reset_index(drop = 1)
    time_idx_len = weather_data_valid.DT.unique().shape[0]
    station_idx_len = weather_data_valid.WBAN.unique().shape[0]

    station_idx = np.sort(weather_data_valid.WBAN.unique())
    time_idx = np.sort(weather_data_valid.DT.unique())
    weather_index = pd.DataFrame(data = {'WBAN': np.repeat(station_idx, time_idx_len),
                                         'DT': np.tile(time_idx, station_idx_len)})
    print('---------------- Construct Wx ----------------')

    TS = construct_wx(weather_index, weather_data_valid, 'TS_mean')
    TS_level = construct_wx(weather_index, weather_data_valid, 'TSlevel_mean')
    Rain = construct_wx(weather_index, weather_data_valid, 'Rain_mean')
    Hail = construct_wx(weather_index, weather_data_valid, 'Hail_mean')
    Precipitation = construct_wx(weather_index, weather_data_valid, 'Precipitation_mean')
    Ice = construct_wx(weather_index, weather_data_valid, 'Ice_mean')
    Squall = construct_wx(weather_index, weather_data_valid, 'Squall_mean')
    Shower = construct_wx(weather_index, weather_data_valid, 'Shower_mean')

    LonLat_Station = valid_station[['LON', 'LAT']].values
    x_ecef, y_ecef, z_ecef = geodetic2ecef(valid_station.LON.values, valid_station.LAT.values)
    TimeIdxTree = cKDTree(time_idx.reshape(-1,1))
    StationIdxTree = cKDTree(np.vstack((x_ecef, y_ecef, z_ecef)).T)
    StationIdxTree_sk = KDTree(np.vstack((x_ecef, y_ecef, z_ecef)).T)
    if Save:
        print('Dumping to file /NOAA/processed_stationwise/')
        try:
            os.mkdir(os.getcwd() + '/NOAA/processed_stationwise/%d'%UTCyear)
        except:
            pass
        weather_data_valid.to_csv(os.getcwd() + '/NOAA/processed_stationwise/%d/valid_station_wx_%d_%s.csv'%(UTCyear, UTCyear,BaseTime.strftime('%Y%m%d%H%M')), index = False)
        pickle.dump(TS, open(os.getcwd() + '/NOAA/processed_stationwise/%d/TS.p'%UTCyear, 'wb'), protocol = 2)
        pickle.dump(TS_level, open(os.getcwd() + '/NOAA/processed_stationwise/%d/TSlevel.p'%UTCyear, 'wb'), protocol = 2)
        pickle.dump(Rain, open(os.getcwd() + '/NOAA/processed_stationwise/%d/Rain.p'%UTCyear, 'wb'), protocol = 2)
        pickle.dump(Hail, open(os.getcwd() + '/NOAA/processed_stationwise/%d/Hail.p'%UTCyear, 'wb'), protocol = 2)
        pickle.dump(Precipitation, open(os.getcwd() + '/NOAA/processed_stationwise/%d/Precipitation.p'%UTCyear, 'wb'), protocol = 2)
        pickle.dump(Ice, open(os.getcwd() + '/NOAA/processed_stationwise/%d/Ice.p'%UTCyear, 'wb'), protocol = 2)
        pickle.dump(Squall, open(os.getcwd() + '/NOAA/processed_stationwise/%d/Squall.p'%UTCyear, 'wb'), protocol = 2)
        pickle.dump(Shower, open(os.getcwd() + '/NOAA/processed_stationwise/%d/Shower.p'%UTCyear, 'wb'), protocol = 2)
        pickle.dump(LonLat_Station, open(os.getcwd() + '/NOAA/processed_stationwise/%d/LonLat_Station.p'%UTCyear, 'wb'), protocol = 2)
        pickle.dump(TimeIdxTree, open(os.getcwd() + '/NOAA/processed_stationwise/%d/TimeIdxTree.p'%UTCyear, 'wb'), protocol = 2)
        pickle.dump(StationIdxTree, open(os.getcwd() + '/NOAA/processed_stationwise/%d/StationIdxTree.p'%UTCyear, 'wb'), protocol = 2)
        pickle.dump(StationIdxTree_sk, open(os.getcwd() + '/NOAA/processed_stationwise/%d/StationIdxTree_sklearn.p'%UTCyear, 'wb'), protocol = 2)
    print('Done!')
    return weather_data_valid, TS, TS_level, Rain, Hail, Precipitation, Ice, Squall, Shower, LonLat_Station, TimeIdxTree, StationIdxTree, StationIdxTree_sk

class WeatherMatching:
    def __init__(self, 
                 DEP, 
                 ARR, 
                 Year, 
                 Wx_load_option = 'Reload', 
                 file_location_dict = None, 
                 VTrack = None,
                 LabelData = None,
                 **kwargs):
        # Options for Wx_load_option: 
        # 'Reload': load weather data from pickle file
        # 'Process': call function ProcessWeatherData to get those data
        global BaseTime
        self.DEP = DEP
        self.ARR = ARR
        self.Year = Year

        if file_location_dict is None:
            file_location_dict  = {'TS':os.getcwd() + '/NOAA/processed_stationwise/' + str(self.Year) + '/TS.p', 
                                   'TSlevel': os.getcwd() + '/NOAA/processed_stationwise/' + str(self.Year) + '/TSlevel.p', 
                                   'Rain': os.getcwd() + '/NOAA/processed_stationwise/' + str(self.Year) + '/Rain.p', 
                                   'Squall': os.getcwd() + '/NOAA/processed_stationwise/' + str(self.Year) + '/Squall.p',
                                   'Hail': os.getcwd() + '/NOAA/processed_stationwise/' + str(self.Year) + '/Hail.p',
                                   'Precipitation': os.getcwd() + '/NOAA/processed_stationwise/' + str(self.Year) + '/Precipitation.p',
                                   'Ice': os.getcwd() + '/NOAA/processed_stationwise/' + str(self.Year) + '/Ice.p',
                                   'Shower': os.getcwd() + '/NOAA/processed_stationwise/' + str(self.Year) + '/Shower.p',
                                   'TimeIdxTree': os.getcwd() + '/NOAA/processed_stationwise/' + str(self.Year) + '/TimeIdxTree.p',
                                   'StationIdxTree': os.getcwd() + '/NOAA/processed_stationwise/' + str(self.Year) + '/StationIdxTree.p',
                                   'StationIdxTree_sk': os.getcwd() + '/NOAA/processed_stationwise/' + str(self.Year) + '/StationIdxTree_sklearn.p',
                                    }
        else:
            file_location_dict = file_location_dict

        if Wx_load_option == 'Reload':
            print('---------------- Preload preprocessed convective weather files ----------------')
            self.TS = pickle.load(open(file_location_dict['TS'], 'rb'))
            self.TSlevel = pickle.load(open(file_location_dict['TSlevel'], 'rb'))
            self.Rain = pickle.load(open(file_location_dict['Rain'], 'rb'))
            self.Squall = pickle.load(open(file_location_dict['Squall'], 'rb'))
            self.Hail = pickle.load(open(file_location_dict['Hail'], 'rb'))
            self.Precipitation = pickle.load(open(file_location_dict['Precipitation'], 'rb'))
            self.Ice = pickle.load(open(file_location_dict['Ice'], 'rb'))
            self.Shower = pickle.load(open(file_location_dict['Shower'], 'rb'))
            self.TimeIdxTree = pickle.load(open(file_location_dict['TimeIdxTree'], 'rb'))
            self.StationIdxTree = pickle.load(open(file_location_dict['StationIdxTree'], 'rb'))
            self.StationIdxTree_sk = pickle.load(open(file_location_dict['StationIdxTree_sk'], 'rb'))
            
            print('Finished')

        elif Wx_load_option == 'Process':
            _, self.TS, self.TS_level, self.Rain, self.Hail, \
               self.Precipitation, self.Ice, self.Squall, self.Shower, \
               _, self.TimeIdxTree, self.StationIdxTree,\
               self.StationIdxTree_sk = ProcessWeatherData(kwargs['Wx_stationwise_dir'], 
                                                           kwargs['Wx_station_dir'], 
                                                           kwargs['UTCyear'],
                                                           kwargs['Save'])

        else:
            raise('ValueError: Wx_load_option is either Reload or Process')

        self.LabelData, self.CenterTraj, self.CenterTraj_ECEF_Coords, self.OutlierTraj, self.OutlierTraj_ECEF_Coords = self.LoadTraj(VTrack,
                                                                                                                                     LabelData)

    def LoadTraj(self, 
                 VTrack = None,
                 LabelData = None):
        print('---------------- Load Trajectories ----------------')
        if VTrack is None:
            VTrackPath = os.getcwd() + '/TFMS_NEW/New_' + self.DEP + self.ARR + str(self.Year) + '.csv'
            self.VTrack = pd.read_csv(VTrackPath)
        else:
            self.VTrack = VTrack
        if LabelData is None:
            LabelData = pd.read_csv(os.getcwd() + '/TFMS_NEW/Label_' + self.DEP+'_' + self.ARR+ '_' + str(self.Year) + '.csv')
        else:
            LabelData = LabelData

        try:
            LabelData.Elap_Time = LabelData.Elap_Time.apply(lambda x: parse(x))
        except:
            pass
        CenterTraj = VTrack[VTrack.FID.isin(LabelData[LabelData.MedianID != -99].FID.values)].reset_index(drop = 1)
        try:
            CenterTraj.Elap_Time = CenterTraj.Elap_Time.apply(lambda x: parse(x))
        except:
            pass

        # Prepare for temporal matching
        CenterTraj['TimeDelta'] = CenterTraj.groupby('FID')['Elap_Time'].transform(lambda x: (x - x.iloc[0]))
        CenterTraj['TimeDelta'] = (CenterTraj['TimeDelta'] - CenterTraj.loc[0,'TimeDelta']).apply(lambda x: x.seconds/3600)

        OutlierTraj = VTrack[VTrack.FID.isin(LabelData[LabelData.ClustID < 0].FID.values)].reset_index(drop = 1)
        # OutlierTraj['TimeDelta'] = OutlierTraj.groupby('FID')['Elap_Time'].transform(lambda x: (x - x.iloc[0]))
        # OutlierTraj['TimeDelta'] = (OutlierTraj['TimeDelta'] - OutlierTraj.loc[0,'TimeDelta']).apply(lambda x: x.seconds/3600)

        x_ecef, y_ecef, z_ecef = geodetic2ecef(CenterTraj.Lon.values, CenterTraj.Lat.values)

        x_ecef_out, y_ecef_out, z_ecef_out = geodetic2ecef(OutlierTraj.Lon.values, OutlierTraj.Lat.values)
        
        print('Finished')
        return LabelData, CenterTraj, np.vstack((x_ecef, y_ecef, z_ecef)).T, OutlierTraj, np.vstack((x_ecef_out, y_ecef_out, z_ecef_out)).T

    def Matching(self, Max_dist = 150, batch_size = 5001):
        # Max_dist in the unit of nautical mile

        # Since this package can only handle one year match, we need a temporary LabelData
        print('Max radius %.1f' %Max_dist)
        tmpLBData = self.LabelData.loc[(self.LabelData.Elap_Time >= parse(str(self.Year)+'-01-01 0:0:0'))&
                                       (self.LabelData.Elap_Time <= parse(str(self.Year)+'-12-31 23:59:59'))].reset_index(drop = True)
        epoch_total = tmpLBData.shape[0]//batch_size
        print('Total Epochs: %d'%(epoch_total+1))
        print('=======================================================')
        self.Max_dist = Max_dist

        # Get spatial query index
        print('---------------- Start spatial matching----------------')
        count_query = self.StationIdxTree_sk.query_radius(self.CenterTraj_ECEF_Coords, \
                                                          r = euclidean_distance(1.852 * Max_dist), \
                                                          count_only=True)
        max_count = np.max(count_query)
        print('\nMax number of stations within the %.2f-nmi-radius around the track point is %d'%(Max_dist, max_count))
        print('\nMin number of stations within the %.2f-nmi-radius around the track point is %d\n'%(Max_dist, np.min(count_query)))
        # construct a matrix with size N center traj * max matched
        StationQueriedIdx = np.zeros((self.CenterTraj_ECEF_Coords.shape[0], max_count), dtype = np.int16)
        Weights = np.zeros((self.CenterTraj_ECEF_Coords.shape[0], max_count))
        StationQueriedIdx_compact, StationDist_compact = self.StationIdxTree_sk.query_radius(self.CenterTraj_ECEF_Coords, \
                                                                      count_only = False, \
                                                                      return_distance = True, \
                                                                      r = euclidean_distance(1.852 * Max_dist)) # nm
        if np.min(count_query) == self.StationIdxTree.data.shape[0]:
            pass
        else:
            for i in range(StationQueriedIdx_compact.shape[0]):
                Len = StationQueriedIdx_compact[i].shape[0]
                StationQueriedIdx[i][:Len] = StationQueriedIdx_compact[i]
                Weights[i][:Len] = 1.0/StationDist_compact[i]
        Weights = Weights/(np.sum(Weights, axis = 1).reshape(-1,1))
        Weights = np.nan_to_num(Weights)

        # release memory...
        del StationQueriedIdx_compact
        del StationDist_compact
        # StationDist, StationQueriedIdx = self.StationIdxTree.query(self.CenterTraj_ECEF_Coords, k = N_point, 
        #                                                                    distance_upper_bound = euclidean_distance(1.852 * Max_dist)) # 60 nm
        # # reset those indices outside of the max distance
        # StationQueriedIdx[StationQueriedIdx == self.StationIdxTree.indices.max() + 1] = 999
        # Weights = (1./StationDist)/(np.sum(1./StationDist, axis = 1).reshape(-1,1))
        # Weights = np.nan_to_num(Weights)

        # Prepare for Matrix product!
        I_matrix = np.zeros((self.CenterTraj.FID.unique().shape[0], self.CenterTraj.shape[0]))
        I_matrix_mean = np.zeros((self.CenterTraj.FID.unique().shape[0], self.CenterTraj.shape[0]))

        for j in range(I_matrix.shape[0]):
            try:
                I_matrix[j, self.CenterTraj.groupby('FID').head(1).index[j]:self.CenterTraj.groupby('FID').head(1).index[j+1]] = 1
                I_matrix_mean[j, self.CenterTraj.groupby('FID').head(1).index[j]:self.CenterTraj.groupby('FID').head(1).index[j+1]] = 1/np.count_nonzero(I_matrix[j,:])
            except:
                I_matrix[j, self.CenterTraj.groupby('FID').head(1).index[j]:] = 1
                I_matrix_mean[j, self.CenterTraj.groupby('FID').head(1).index[j]:] = 1/np.count_nonzero(I_matrix[j,:])

        def FinializeMatching(Wx, SpatialIdx, WeightsIdx, TimeQuery):
            matched_Wx = Wx[TimeQueriedIdx.reshape(-1,1), SpatialIdx]
            matched_Wx_weighted_sum = np.sum(np.multiply(matched_Wx, WeightsIdx), axis = 1).reshape(TimeQuery.shape)
            Nominal_matched_Wx = I_matrix_mean.dot(matched_Wx_weighted_sum.T).T.reshape(-1,1)
            return Nominal_matched_Wx

        print('---------------- Start temporal matching ----------------')
        
        self.MemberFID = []
        st = time.time()

        for epoch in range(epoch_total + 1):
            print('Batch %d' %(epoch+1))
            st_epoch = time.time()
            
            TimeQuery = []
            # For each batch, get the departure time index
            for i in range(batch_size * epoch, min(tmpLBData.shape[0], batch_size * (epoch + 1))):
                self.MemberFID.append(tmpLBData.loc[i, 'FID'])
                departureTime = tmpLBData.loc[i, 'Elap_Time']
                dt = departureTime - BaseTime
                dt = dt.total_seconds()/3600.
                TimeQuery.append(self.CenterTraj.TimeDelta.values + dt)
            TimeQuery = np.array(TimeQuery)
            TimeDist, TimeQueriedIdx = self.TimeIdxTree.query(TimeQuery.reshape(-1,1))
            TimeQueriedIdx = TimeQueriedIdx.reshape(TimeQuery.shape) # n flights * m nominal routes

            print('Start Heavy Duty!!! Memory EXPLODEs sometimes!!! If so, please try smaller batch size!')
            SpatialIdx = np.zeros(shape = (StationQueriedIdx.shape[0] * TimeQueriedIdx.shape[0], max_count), dtype = np.int16)
            WeightsIdx = np.zeros(shape = (StationQueriedIdx.shape[0] * TimeQueriedIdx.shape[0], max_count), dtype = np.float32)
            for i in range(max_count):
                SpatialIdx[:,i] = np.tile(StationQueriedIdx[:,i], TimeQueriedIdx.shape[0])
                WeightsIdx[:,i] = np.tile(Weights[:,i], TimeQueriedIdx.shape[0])
            print('Shape of Spatial Index Matrix: ', SpatialIdx.shape)
            print('Shape of Weight Index Matrix (should be agree with the spatial index): ', WeightsIdx.shape)
            print('Shape of Temporal Index Matrix: ', TimeQueriedIdx.shape)
            print('---------------- Finilize matching and reshaping ----------------')

            if epoch == 0:

                self.Nominal_matched_TS = FinializeMatching(self.TS, SpatialIdx, WeightsIdx, TimeQuery)
                self.Nominal_matched_TSlevel = FinializeMatching(self.TSlevel, SpatialIdx, WeightsIdx, TimeQuery)
                self.Nominal_matched_Rain = FinializeMatching(self.Rain, SpatialIdx, WeightsIdx, TimeQuery)
                self.Nominal_matched_Hail = FinializeMatching(self.Hail, SpatialIdx, WeightsIdx, TimeQuery)
                self.Nominal_matched_Precipitation = FinializeMatching(self.Precipitation, SpatialIdx, WeightsIdx, TimeQuery)
                self.Nominal_matched_Shower = FinializeMatching(self.Shower, SpatialIdx, WeightsIdx, TimeQuery)
                self.Nominal_matched_Ice = FinializeMatching(self.Ice, SpatialIdx, WeightsIdx, TimeQuery)
                self.Nominal_matched_Squall = FinializeMatching(self.Squall, SpatialIdx, WeightsIdx, TimeQuery)

            else:
                self.Nominal_matched_TS = np.append(self.Nominal_matched_TS, FinializeMatching(self.TS, SpatialIdx, WeightsIdx, TimeQuery), axis = 0)
                self.Nominal_matched_TSlevel = np.append(self.Nominal_matched_TSlevel, FinializeMatching(self.TSlevel, SpatialIdx, WeightsIdx, TimeQuery), axis = 0)
                self.Nominal_matched_Rain = np.append(self.Nominal_matched_Rain, FinializeMatching(self.Rain, SpatialIdx, WeightsIdx, TimeQuery), axis = 0)
                self.Nominal_matched_Hail = np.append(self.Nominal_matched_Hail, FinializeMatching(self.Hail, SpatialIdx, WeightsIdx, TimeQuery), axis = 0)
                self.Nominal_matched_Precipitation = np.append(self.Nominal_matched_Precipitation, FinializeMatching(self.Precipitation, SpatialIdx, WeightsIdx, TimeQuery), axis = 0)
                self.Nominal_matched_Shower = np.append(self.Nominal_matched_Shower, FinializeMatching(self.Shower, SpatialIdx, WeightsIdx, TimeQuery), axis = 0)
                self.Nominal_matched_Ice = np.append(self.Nominal_matched_Ice, FinializeMatching(self.Ice, SpatialIdx, WeightsIdx, TimeQuery), axis = 0)
                self.Nominal_matched_Squall = np.append(self.Nominal_matched_Squall, FinializeMatching(self.Squall, SpatialIdx, WeightsIdx, TimeQuery), axis = 0)
            print('Epoch Time: %.2f' %(time.time() - st_epoch))
            print('---------------- Finished Matching for Batch %d----------------'%(epoch + 1))
        print('*******=====================Finished========================*******')
        self.MemberFID = np.array(self.MemberFID)

        return self.Nominal_matched_TS, self.Nominal_matched_TSlevel, self.Nominal_matched_Rain, \
               self.Nominal_matched_Hail, self.Nominal_matched_Precipitation, self.Nominal_matched_Squall, \
               self.Nominal_matched_Shower, self.Nominal_matched_Ice

    def MergeWithMNL(self, 
                     TimeZone, 
                     Save = True,
                     MNL_Generic_name = None,
                     MNLWX_outname = None
                     ):
        # If Generic MNL exists, load. Otherwise, build
        # TimeZone is for Departure airport
        MNL_Generic_name = MNL_Generic_name or (os.getcwd() + '/MNL/MNL_Generic/' + self.DEP + '_' + self.ARR + '_' + str(self.Year) + '.csv')
        MNLWX_outname = MNLWX_outname or (os.getcwd() + '/MNL/WSx_MNL_' + self.DEP + self.ARR + '_' + str(self.Year) + '_' + str(self.Max_dist) + '.csv')
        if os.path.isfile(MNL_Generic_name):
            MNL = pd.read_csv(MNL_Generic_name)
        else:
            from GetMNLdataset_generic import GetMNLdataset
            MNL_class = GetMNLdataset(self.DEP, 
                                      self.ARR, 
                                      self.Year, 
                                      TimeZone,
                                      self.VTrack,
                                      self.LabelData)
            MNL = MNL_class.ConstructingMNL(Save = Save,
                                            output_dir = MNL_Generic_name)

        MergingIdx = np.hstack((np.repeat(self.CenterTraj.groupby('FID').head(1).FID.values.reshape(1,-1), self.MemberFID.shape[0], axis = 0).reshape(-1,1), 
                                np.repeat(self.MemberFID.reshape(-1,1), self.CenterTraj.FID.unique().shape[0], axis = 0)))
        Wx_df = pd.DataFrame(MergingIdx, columns=['NominalFID', 'MemberFID'])
        Wx_df['TS_mean'] = self.Nominal_matched_TS
        Wx_df['TSlevel_mean'] = self.Nominal_matched_TSlevel
        Wx_df['Rain_mean'] = self.Nominal_matched_Rain
        Wx_df['Hail_mean'] = self.Nominal_matched_Hail
        Wx_df['Precipitation_mean'] = self.Nominal_matched_Precipitation
        Wx_df['Shower_mean'] = self.Nominal_matched_Shower
        Wx_df['Ice_mean'] = self.Nominal_matched_Ice
        Wx_df['Squall_mean'] = self.Nominal_matched_Squall

        MNL = MNL.merge(Wx_df, left_on=['FID_x', 'FID_Member'], right_on=['NominalFID', 'MemberFID'], how = 'left')
        del MNL['NominalFID']
        del MNL['MemberFID']

        # Wx_out_df = self.OutlierTraj[['FID']]
        # Wx_out_df['TS_mean'] = self.Nominal_matched_TS_out
        # Wx_out_df['TSlevel_mean'] = self.Nominal_matched_TSlevel_out
        # Wx_out_df['Rain_mean'] = self.Nominal_matched_Rain_out
        # Wx_out_df['Hail_mean'] = self.Nominal_matched_Hail_out
        # Wx_out_df['Precipitation_mean'] = self.Nominal_matched_Precipitation_out
        # Wx_out_df['Shower_mean'] = self.Nominal_matched_Shower_out
        # Wx_out_df['Ice_mean'] = self.Nominal_matched_Ice_out
        # Wx_out_df['Squall_mean'] = self.Nominal_matched_Squall_out

        if Save:
            MNL.to_csv(MNLWX_outname, index = False)
        return MNL

    # def MatchingOutlier(self, N_point = 20, Max_dist = 60):

    #     print('---------------- Prepare for temporal matching ----------------')
    #     TimeQuery = (self.OutlierTraj['Elap_Time'] - BaseTime).apply(lambda x: x.total_seconds()/3600.).values
    #     print('---------------- Start temporal matching----------------')
    #     TimeDist, TimeQueriedIdx = self.TimeIdxTree.query(TimeQuery.reshape(-1,1))
    #     TimeQueriedIdx = TimeQueriedIdx.reshape(TimeQuery.shape) # n flights * m nominal routes
    #     print('---------------- Finished----------------')

    #     print('---------------- Start spatial matching----------------')
    #     StationDist, StationQueriedIdx = self.StationIdxTree.query(self.OutlierTraj_ECEF_Coords, k = N_point, 
    #                                                                        distance_upper_bound = euclidean_distance(1.852 * Max_dist)) # 60 nm
    #     # reset those indices outside of the max distance
    #     StationQueriedIdx[StationQueriedIdx == self.StationIdxTree.indices.max() + 1] = 999
    #     Weights = (1./StationDist)/(np.sum(1./StationDist, axis = 1).reshape(-1,1))
    #     Weights = np.nan_to_num(Weights)

    #     # Structure the queried indices
    #     SpatialIdx = StationQueriedIdx
    #     WeightsIdx = Weights

    #     print('Shape of Spatial Index Matrix: ', SpatialIdx.shape)
    #     print('Shape of Weight Index Matrix (should be agree with the spatial index): ', WeightsIdx.shape)
    #     print('Shape of Temporal Index Matrix: ', TimeQueriedIdx.shape)
        
    #     print('---------------- Finilize matching and reshaping ----------------')

    #     I_matrix = np.zeros((self.OutlierTraj.FID.unique().shape[0], self.OutlierTraj.shape[0]))
    #     I_matrix_mean = np.zeros((self.OutlierTraj.FID.unique().shape[0], self.OutlierTraj.shape[0]))

    #     for j in range(I_matrix.shape[0]):
    #         try:
    #             I_matrix[j, self.OutlierTraj.groupby('FID').head(1).index[j]:self.OutlierTraj.groupby('FID').head(1).index[j+1]] = 1
    #             I_matrix_mean[j, self.OutlierTraj.groupby('FID').head(1).index[j]:self.OutlierTraj.groupby('FID').head(1).index[j+1]] = 1/np.count_nonzero(I_matrix[j,:])
    #         except:
    #             I_matrix[j, self.OutlierTraj.groupby('FID').head(1).index[j]:] = 1
    #             I_matrix_mean[j, self.OutlierTraj.groupby('FID').head(1).index[j]:] = 1/np.count_nonzero(I_matrix[j,:])

    #     def FinializeMatching(Wx):
    #         matched_Wx = Wx[TimeQueriedIdx.reshape(-1,1), SpatialIdx]
    #         matched_Wx_weighted_sum = np.sum(np.multiply(matched_Wx, WeightsIdx), axis = 1).reshape(TimeQuery.shape)
    #         Nominal_matched_Wx = I_matrix_mean.dot(matched_Wx_weighted_sum.T).T.reshape(-1,1)
    #         return Nominal_matched_Wx

    #     self.Nominal_matched_TS_out = FinializeMatching(self.TS)
    #     self.Nominal_matched_TSlevel_out = FinializeMatching(self.TSlevel)
    #     self.Nominal_matched_Rain_out = FinializeMatching(self.Rain)
    #     self.Nominal_matched_Hail_out = FinializeMatching(self.Hail)
    #     self.Nominal_matched_Precipitation_out = FinializeMatching(self.Precipitation)
    #     self.Nominal_matched_Shower_out = FinializeMatching(self.Shower)
    #     self.Nominal_matched_Ice_out = FinializeMatching(self.Ice)
    #     self.Nominal_matched_Squall_out = FinializeMatching(self.Squall)
    #     print('*******---------------------Finished------------------------*******')

    #     return self.Nominal_matched_TS_out, self.Nominal_matched_TSlevel_out, self.Nominal_matched_Rain_out, \
    #            self.Nominal_matched_Hail_out, self.Nominal_matched_Precipitation_out, self.Nominal_matched_Squall_out, \
    #            self.Nominal_matched_Shower_out, self.Nominal_matched_Ice_out