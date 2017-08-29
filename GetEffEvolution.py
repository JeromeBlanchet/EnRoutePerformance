# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 2017

@ Author: Liu, Yulin
@ Institute: UC Berkeley
"""

from __future__ import division
import os
import GetCluster
import GetEDA
import numpy as np


class EffEvolution:
    def __init__(self, Dep = 'LAX', Arr = 'SEA', Timeframe = [2013, 2014, 2015]):
        self.Dep = Dep
        self.Arr = Arr
        self.Timeframe = Timeframe

    def CleanData(self, InputData = False, SaveTrack = True, **kwargs):
        A_cutoff = kwargs.get('Cutoff', 0.5)
        T_cutoff = kwargs.get('Tcut', 300)
        D_cutoff = kwargs.get('Dcut', 100)
        V_cutoff = kwargs.get('Vcut', 0.27)
        for year in self.Timeframe:
            print('Processing flights from %s to %s in %d'%(self.Dep, self.Arr, year))
            exec("self.Dep_Arr_%s = GetEDA.EDA_Data(self.Dep, self.Arr, year, A_cutoff, T_cutoff, D_cutoff, V_cutoff, InputData = InputData, db = False, Insert = False)"%str(year))
            if SaveTrack:
                print('Saving flights from %s to %s in %d'%(self.Dep, self.Arr, year))
                exec("self.Dep_Arr_%s.SaveData()"%str(year))
        return

    def MergeData(self):
        i = 0
        for year in self.Timeframe:
            if i == 0:
                exec("self.All_Eff = self.Dep_Arr_%s.Efficiency.copy()"%str(year))
                exec("self.All_VTrack = self.Dep_Arr_%s.VTrack.copy()"%str(year))
                i += 1
            else:
                exec("self.All_Eff.update(self.Dep_Arr_%s.Efficiency)"%str(year))
                exec("self.All_VTrack = self.All_VTrack.append(self.Dep_Arr_%s.VTrack)"%str(year))
                i += 1
        self.All_VTrack = self.All_VTrack.reset_index(drop = True)
        print('Number of flights with the specified time frame: ', self.All_VTrack.FID.unique().shape)

        return self.All_VTrack, self.All_Eff

    def Pre_Clustering(self, N_Comp = 5, N_pt = 100):
        self.T1 = GetCluster.Traj_Clustering(self.Dep,self.Arr, 9999, N_Comp = N_Comp, N_pt = N_pt, VTRACK = self.All_VTrack, EnEff = self.All_Eff)
    def Clustering(self, dist_thres = 1, num_thres = 20, **kwargs):
        SaveLabelData = kwargs.get('SaveData', False)
        Median = kwargs.get('MEDIAN', True)
        Plot = kwargs.get('PLOT', True)
        LBdata1, _ = self.T1.DB_Clustering(dist_thres, num_thres, SAVE = SaveLabelData, MEDIAN = Median, PLOT = Plot)

        stat_summary = LBdata1.groupby(['YEAR','ClustID']).agg({'FID': np.count_nonzero, 'Efficiency': np.mean}).reset_index()
        stat_summary['share'] = stat_summary.groupby('YEAR').FID.transform(lambda x: x/x.sum())
        stat_summary = stat_summary.merge(LBdata1.groupby('ClustID').Efficiency.mean().reset_index(), on = 'ClustID')
        stat_summary.columns = ['Year', 'ClusterID', 'WithinClusterInefficiency', 'TotalTraffic', 'ShareOfTraffic','ClusterAverageIneff']

        return LBdata1, stat_summary[['Year', 'ClusterID','ClusterAverageIneff', 'WithinClusterInefficiency', 'TotalTraffic', 'ShareOfTraffic']].sort_values(by = ['Year', 'ClusterID'])