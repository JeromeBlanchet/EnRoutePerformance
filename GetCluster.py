# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 09:51:38 2016

@ Author: Liu, Yulin
@ Institute: UC Berkeley
"""

import os
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pickle
from sklearn import cluster
from scipy.spatial.distance import pdist, squareform
from GetEDA import GetCoords, Basemap
from GetPreferRoute import PreferRoute
from KMedoids import KMedoids

class Traj_Clustering:
    def __init__(self,
                 Ori, 
                 Des, 
                 year, 
                 N_pt = 100, 
                 N_Comp = 4,
                 altitude = False, 
                 VTRACK = None, 
                 EnEff = None,
                 Prefer = False, 
                 **kwargs):
        self.Ori = Ori
        self.Des = Des
        self.year = year
        self.altitude = altitude
        self.Coords = GetCoords()
        self.Ori_Lat = self.Coords[self.Ori][0]
        self.Ori_Lon = self.Coords[self.Ori][1]
        self.Des_Lat = self.Coords[self.Des][0]
        self.Des_Lon = self.Coords[self.Des][1]
        
        if VTRACK is None:
            try:
                print('Infer data location')
                self.VTRACK = pd.read_csv(os.getcwd()+'\TFMS_NEW\\'+'New_'+self.Ori+self.Des+str(self.year)+'.csv')
            except:
                raise ValueError('No trajectory data found! Please load valid flight track data!')
#                DATA = EDA_Data(self.Ori,self.Des,self.year)
#                self.VTRACK = DATA.VTrack
        else:
            print('Load external VTRACK data')
            self.VTRACK = VTRACK.copy()
        self.XLIMU = kwargs.get('xrb', self.VTRACK.Lon.max()+1)
        self.XLIML = kwargs.get('xlb', self.VTRACK.Lon.min()-1)
        self.YLIMU = kwargs.get('yub', self.VTRACK.Lat.max()+1)
        self.YLIML = kwargs.get('ylb', self.VTRACK.Lat.min()-1)
        if EnEff is None:
            try:            
                EnEff = pickle.load(open(os.getcwd()+'\TFMS_NEW\\'+'Eff_'+self.Ori+self.Des+str(self.year)+'.p','rb'))
            except:
                raise ValueError('No Efficiency data found! Please load valid dataset!')
        else:
            EnEff = EnEff.copy()
        self.EnEff = pd.DataFrame(EnEff).T.reset_index(drop = False)
        self.EnEff.columns = ['FID','D40A100ACT','D40A100ACH','Efficiency']
        
        self.FID_ID = self.VTRACK.FID.unique()
        self.n_flights = self.FID_ID.shape[0]
        
        self.N_pt = N_pt
        self.N_Comp = N_Comp
        
        self.NewSeries = self.Interpolation()
        self.PCA_Inv, self.PCA_Comp, self.pca, self._mu, self._std = self.PCA_Traj()
        
        if Prefer == True:
            KEY1 = self.Ori+'_'+self.Des
            KEY2 = 'K'+self.Ori+'_K'+self.Des
            self.PR = PreferRoute()
            try:
                self.PrefRoute = self.PR.Pref_Route[KEY1]
            except:
                self.PrefRoute = {}
            try:
                self.CDRoute = self.PR.CD_Route[KEY2]
            except:
                self.CDRoute = {}
        
        # Plotting
        
    def Interpolation(self):
        NewSeries = []
        OldSeries = []    
        i = 0
        for idx, gp in self.VTRACK.groupby('FID'):
            i += 1
            # Interpolated in terms of distance traveled
            dold = gp.CumDist.values
            xold = gp.Lon.values
            yold = gp.Lat.values
            f1 = interp1d(dold,xold,kind = 'linear')
            f2 = interp1d(dold,yold,kind = 'linear')
            dnew = np.linspace(dold[0],dold[-1],num = self.N_pt, endpoint = True)
            xnew = f1(dnew)
            ynew = f2(dnew)
            
            if self.altitude:
                zold = gp.Alt.values
                f3 = interp1d(dold,zold,kind = 'linear')
                znew = f3(dnew)
                OldSeries.append(np.concatenate((xold,yold,zold)))
                NewSeries.append(np.concatenate((xnew,ynew,znew)))
            else:
                OldSeries.append(np.concatenate((xold,yold)))
                NewSeries.append(np.concatenate((xnew,ynew)))
        NewSeries = np.asarray(NewSeries)
        OldSeries = np.asarray(OldSeries)
        return NewSeries
        
    def PCA_Traj(self):
        
        _mu, _std = np.mean(self.NewSeries, axis = 0), np.std(self.NewSeries, axis = 0)
        pca = PCA(n_components = self.N_Comp)
        PCA_Compon = pca.fit_transform((self.NewSeries - _mu)/_std)
        print(pca.explained_variance_ratio_)
        print(pca.explained_variance_)
        print(sum(pca.explained_variance_ratio_))
        PCA_Inv = pca.inverse_transform(PCA_Compon)
        return (PCA_Inv*_std+_mu),PCA_Compon,pca, _mu, _std
    
    def Plot_PCA_Int(self,xlb = -126,xrb = -80, ylb = 26,yub = 50):
        self.XLIMU = xrb
        self.XLIML = xlb
        self.YLIMU = yub
        self.YLIML = ylb
        if self.altitude:
            N_pnt = self.NewSeries.shape[1]//3
        else:
            N_pnt = self.NewSeries.shape[1]//2
        
        fig = plt.figure(figsize=(16,6))
        if not self.altitude:
            ax1 = fig.add_subplot(121)
            ax2 = fig.add_subplot(122,sharex=ax1,sharey=ax1)
        else:
            ax1 = fig.add_subplot(221)
            ax2 = fig.add_subplot(222,sharex=ax1,sharey=ax1)
            ax3 = fig.add_subplot(223)
            ax4 = fig.add_subplot(224,sharex=ax3,sharey=ax3)

        ax1.scatter(self.NewSeries[:,:N_pnt],self.NewSeries[:,N_pnt:2*N_pnt],edgecolor = 'b', s = 0.01, label = 'Interpolated')
        ax2.scatter(self.PCA_Inv[:,:N_pnt],self.PCA_Inv[:,N_pnt:2*N_pnt],edgecolor = 'g',s = 0.01, label = 'Inverse PCA')
        ax1.set_xlim(self.XLIML,self.XLIMU)
        ax1.set_ylim(self.YLIML,self.YLIMU)
        ax1.set_title('Original Resampled Trajectories')
        ax1.set_xlabel('Lon')
        ax1.set_ylabel('Lat')
        ax2.set_title('Trajectories based on PCA with %d factors'%self.N_Comp)
        ax2.set_xlabel('Lon')
        ax2.set_ylabel('Lat')

        if self.altitude:
            ax3.scatter(np.tile(range(self.N_pt), self.n_flights), self.NewSeries[:,2*N_pnt:3*N_pnt],edgecolor = 'b', s = 0.01, label = 'Interpolated')
            ax4.scatter(np.tile(range(self.N_pt), self.n_flights), self.PCA_Inv[:,2*N_pnt:3*N_pnt],edgecolor = 'g',s = 0.01, label = 'Inverse PCA')
            ax3.set_xlabel('n point')
            ax3.set_ylabel('Altitude (100 ft)')
            ax4.set_xlabel('n point')
            ax4.set_ylabel('Altitude (100 ft)')
    
    def Dist_Mat(self,x, Weighted = False, weights = None):
        if Weighted:
            return squareform(pdist(x,'seuclidean', V = weights))
        else:
            return squareform(pdist(x,'euclidean'))
    
    def NominalRoute(self,x, Weighted = False, weights = None):
        Dist_matrix = self.Dist_Mat(x, Weighted, weights)
        TDist = sum(Dist_matrix)
        Idx = np.argmin(TDist)
        return x[Idx], self.FID_ID[np.where(self.PCA_Comp == x[Idx])[0][0]]
    
    def DecomposeOutlier(self, LBData = None, n_rep_outlier = 3, SAVE = False):
        if LBData is None:
            LabelData = self.LBData
        else:
            LabelData = LBData.copy()
        outlier_fid = LabelData.loc[LabelData.ClustID == -1, 'FID'].values
        T1_out = Traj_Clustering(self.Ori, self.Des, self.year, 
                                 VTRACK = self.VTRACK.loc[self.VTRACK.FID.isin(outlier_fid)], 
                                 N_Comp = self.N_Comp,
                                 altitude = self.altitude)
        LBdata_outlier, fig_outlier = T1_out.Kmeans_Clustering(n_clust = n_rep_outlier, SAVE=False, MEDIAN=True, PLOT = True, QuickDraw=False)
        LBdata_outlier['ClustID'] = LBdata_outlier['ClustID'].apply(lambda x: -1 - x)
        LBdata_outlier['MedianID'] = LBdata_outlier['MedianID'].apply(lambda x: int(x*(x==-99) + (-1-x)*(x!=-99)))
        LBdata_all = LabelData.loc[~LabelData.FID.isin(outlier_fid),:].append(LBdata_outlier).reset_index(drop = True)
        if SAVE:
            Fname = 'Label_'+self.Ori+'_'+self.Des+'_'+str(self.year)
            LBdata_all.to_csv(os.getcwd()+'\TFMS_NEW\\'+Fname+'.csv', index = False)
            # pickle.dump(Median_ID, open(os.getcwd()+'\TFMS_NEW\\MEDIAN_'+Fname+'.p','wb'))

        return LBdata_outlier, LBdata_all, fig_outlier


    def DB_Clustering(self, 
                      EPS, 
                      Min_Samp, 
                      Weighted = False, 
                      PLOT = True, 
                      MEDIAN = False, 
                      Prefer = False, 
                      SAVE = False, 
                      output_dir = None,
                      QuickDraw = True, 
                      verbose = False):
        self.EPS = EPS
        self.Min_Samp = Min_Samp
        self.Weighted = Weighted

        if Weighted:
            dbscan = cluster.DBSCAN(eps = EPS, min_samples = Min_Samp, metric = 'precomputed')
            WeightedDist = self.Dist_Mat(self.PCA_Comp, Weighted = True, weights = 1./self.pca.explained_variance_ratio_)
            db = dbscan.fit(WeightedDist)
        else:
            dbscan = cluster.DBSCAN(eps = EPS, min_samples = Min_Samp, metric = 'euclidean')
            db = dbscan.fit(self.PCA_Comp)
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        labels = db.labels_
        
        ClusterIDdata = pd.DataFrame(np.dstack([self.FID_ID,labels])[0],columns=['FID','ClustID'])
        LabelData = self.VTRACK.groupby('FID').head(1).ix[:,:7].reset_index(drop = 1).merge(self.VTRACK.groupby('FID').DT.sum().reset_index(), left_on='FID', right_on='FID', how='left')
        
        LabelData = LabelData.merge(ClusterIDdata,left_on = 'FID',right_on = 'FID').merge(self.EnEff,left_on = 'FID',right_on = 'FID',how = 'left')
        
        # Number of clusters in labels, ignoring noise if present.
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        print('Estimated number of clusters: %d' % n_clusters_)
        
        # PLOTTING
        if PLOT == True:
            Median_ID, fig = self.plot_clusters(labels, QuickDraw, MEDIAN, LabelData, verbose)
        
        LabelData['YEAR'] = LabelData.Elap_Time.apply(lambda x: int(x[:4]))
        LabelData['MONTH'] = LabelData.Elap_Time.apply(lambda x: int(x[5:7]))
        LabelData['DAY'] = LabelData.Elap_Time.apply(lambda x: int(x[8:10]))
        LabelData['Hour'] = LabelData.Elap_Time.apply(lambda x: int(x[11:13]))
        if MEDIAN:
            LabelData['MedianID'] =  LabelData.FID.apply(lambda x: Median_ID[x] if x in Median_ID else -99)    # -99 IF NOT MEDIAN
        self.LBData = LabelData
        if SAVE == True:
            if output_dir is None:
                Fname = 'Label_'+self.Ori+'_'+self.Des+'_'+str(self.year)
            else:
                Fname = output_dir
            LabelData.to_csv(os.getcwd()+'/TFMS_NEW/'+Fname+'.csv', index = False)
            pickle.dump(Median_ID, open(os.getcwd()+'/TFMS_NEW/MEDIAN_'+Fname+'.p','wb'))
            print('file saved to %s'%(os.getcwd()+'/TFMS_NEW/'+Fname+'.csv'))
        else:
            pass
        if PLOT:
            return LabelData, fig
        else:
            return LabelData, SAVE
            
    def Kmeans_Clustering(self, n_clust, PLOT = True, MEDIAN = False, Prefer = False, SAVE = False, QuickDraw = True, verbose = False):
        kmeans = cluster.KMeans(n_clusters = n_clust, init = 'k-means++', random_state = 101).fit(self.PCA_Comp)
        labels = kmeans.labels_
        
        ClusterIDdata = pd.DataFrame(np.dstack([self.FID_ID,labels])[0],columns=['FID','ClustID'])
        LabelData = self.VTRACK.groupby('FID').head(1).ix[:,:7].reset_index(drop = 1).merge(self.VTRACK.groupby('FID').DT.sum().reset_index(), left_on='FID', right_on='FID', how='left')
        
        LabelData = LabelData.merge(ClusterIDdata,left_on = 'FID',right_on = 'FID').merge(self.EnEff,left_on = 'FID',right_on = 'FID',how = 'left')
        
        # Number of clusters in labels, ignoring noise if present.
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        print('Estimated number of clusters: %d' % n_clusters_)
        
        # PLOTTING
        if PLOT == True:
            Median_ID, fig = self.plot_clusters(labels, QuickDraw, MEDIAN, LabelData, verbose)
        try:
            LabelData['YEAR'] = LabelData.Elap_Time.apply(lambda x: int(x[:4]))
            LabelData['MONTH'] = LabelData.Elap_Time.apply(lambda x: int(x[5:7]))
            LabelData['DAY'] = LabelData.Elap_Time.apply(lambda x: int(x[8:10]))
            LabelData['Hour'] = LabelData.Elap_Time.apply(lambda x: int(x[11:13]))
        except:
            LabelData['YEAR'] = LabelData.Elap_Time.apply(lambda x: x.year)
            LabelData['MONTH'] = LabelData.Elap_Time.apply(lambda x: x.month)
            LabelData['DAY'] = LabelData.Elap_Time.apply(lambda x: x.day)
            LabelData['Hour'] = LabelData.Elap_Time.apply(lambda x: x.hour)
        if MEDIAN:
            LabelData['MedianID'] =  LabelData.FID.apply(lambda x: Median_ID[x] if x in Median_ID else -99)    # -99 IF NOT MEDIAN
        
        if SAVE == True:
            Fname = 'Label_'+self.Ori+'_'+self.Des+'_'+str(self.year)
            LabelData.to_csv(os.getcwd()+'\TFMS_NEW\\'+Fname+'.csv', index = False)
            pickle.dump(Median_ID, open(os.getcwd()+'\TFMS_NEW\\MEDIAN_'+Fname+'.p','wb'))
        else:
            pass
        if PLOT:
            return LabelData, fig
        else:
            return LabelData, SAVE

    def KMedoids_Clustering(self, n_clust, PLOT = True, MEDIAN = False, Prefer = False, SAVE = False, QuickDraw = True, verbose = False):
        labels = -1*np.ones((self.PCA_Comp.shape[0],), dtype=np.int8)
        kmedoids = KMedoids(n_cluster = n_clust, max_iter = 1000)
        kmedoids.fit(self.PCA_Comp)
        print(kmedoids.clusters.keys())
        
        _i = 0
        for _key in kmedoids.clusters:
            labels[kmedoids.clusters[_key]] = _i
            _i += 1
        
        ClusterIDdata = pd.DataFrame(np.dstack([self.FID_ID,labels])[0],columns=['FID','ClustID'])
        LabelData = self.VTRACK.groupby('FID').head(1).ix[:,:7].reset_index(drop = 1).merge(self.VTRACK.groupby('FID').DT.sum().reset_index(), left_on='FID', right_on='FID', how='left')
        
        LabelData = LabelData.merge(ClusterIDdata,left_on = 'FID',right_on = 'FID').merge(self.EnEff,left_on = 'FID',right_on = 'FID',how = 'left')
        
        # Number of clusters in labels, ignoring noise if present.
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        print('Estimated number of clusters: %d' % n_clusters_)
        
        # PLOTTING
        if PLOT == True:
            Median_ID, fig = self.plot_clusters(labels, QuickDraw, MEDIAN, LabelData, verbose)
        try:
            LabelData['YEAR'] = LabelData.Elap_Time.apply(lambda x: int(x[:4]))
            LabelData['MONTH'] = LabelData.Elap_Time.apply(lambda x: int(x[5:7]))
            LabelData['DAY'] = LabelData.Elap_Time.apply(lambda x: int(x[8:10]))
            LabelData['Hour'] = LabelData.Elap_Time.apply(lambda x: int(x[11:13]))
        except:
            LabelData['YEAR'] = LabelData.Elap_Time.apply(lambda x: x.year)
            LabelData['MONTH'] = LabelData.Elap_Time.apply(lambda x: x.month)
            LabelData['DAY'] = LabelData.Elap_Time.apply(lambda x: x.day)
            LabelData['Hour'] = LabelData.Elap_Time.apply(lambda x: x.hour)
        if MEDIAN:
            LabelData['MedianID'] =  LabelData.FID.apply(lambda x: Median_ID[x] if x in Median_ID else -99)    # -99 IF NOT MEDIAN
        
        if SAVE == True:
            Fname = 'Label_'+self.Ori+'_'+self.Des+'_'+str(self.year)
            LabelData.to_csv(os.getcwd()+'\TFMS_NEW\\'+Fname+'.csv', index = False)
            pickle.dump(Median_ID, open(os.getcwd()+'\TFMS_NEW\\MEDIAN_'+Fname+'.p','wb'))
        else:
            pass
        if PLOT:
            return LabelData, fig
        else:
            return LabelData, SAVE
            
    def Spectral_Clustering(self, n_clust, PLOT = True, MEDIAN = False, Prefer = False, SAVE = False, QuickDraw = True):
        spectral = cluster.SpectralClustering(n_clusters = n_clust, gamma = 1, random_state = 101).fit(self.PCA_Comp)
        labels = spectral.labels_
        
        ClusterIDdata = pd.DataFrame(np.dstack([self.FID_ID,labels])[0],columns=['FID','ClustID'])
        LabelData = self.VTRACK.groupby('FID').head(1).ix[:,:7].reset_index(drop = 1).merge(self.VTRACK.groupby('FID').DT.sum().reset_index(), left_on='FID', right_on='FID', how='left')
        
        LabelData = LabelData.merge(ClusterIDdata,left_on = 'FID',right_on = 'FID').merge(self.EnEff,left_on = 'FID',right_on = 'FID',how = 'left')
        
        # Number of clusters in labels, ignoring noise if present.
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        print('Estimated number of clusters: %d' % n_clusters_)
        
        # PLOTTING
        if PLOT == True:
            Median_ID, fig = self.plot_clusters(labels, QuickDraw, MEDIAN, LabelData, verbose)
        
        LabelData['YEAR'] = LabelData.Elap_Time.apply(lambda x: int(x[:4]))
        LabelData['MONTH'] = LabelData.Elap_Time.apply(lambda x: int(x[5:7]))
        LabelData['DAY'] = LabelData.Elap_Time.apply(lambda x: int(x[8:10]))
        LabelData['Hour'] = LabelData.Elap_Time.apply(lambda x: int(x[11:13]))
        if MEDIAN:
            LabelData['MedianID'] =  LabelData.FID.apply(lambda x: Median_ID[x] if x in Median_ID else -99)    # -99 IF NOT MEDIAN
        
        if SAVE == True:
            Fname = 'Label_'+self.Ori+'_'+self.Des+'_'+str(self.year)
            LabelData.to_csv(os.getcwd()+'\TFMS_NEW\\'+Fname+'.csv', index = False)
            pickle.dump(Median_ID, open(os.getcwd()+'\TFMS_NEW\\MEDIAN_'+Fname+'.p','wb'))
        else:
            pass
        if PLOT:
            return LabelData, fig
        else:
            return LabelData, SAVE
            
    def MS_Clustering(self, bw, PLOT = True, MEDIAN = False, Prefer = False, SAVE = False, QuickDraw = True):
        ms = cluster.MeanShift(bandwidth=bw, bin_seeding=True, cluster_all = False).fit(self.PCA_Comp)

        labels = ms.labels_
        cluster_centers = ms.cluster_centers_
        
        ClusterIDdata = pd.DataFrame(np.dstack([self.FID_ID,labels])[0],columns=['FID','ClustID'])
        LabelData = self.VTRACK.groupby('FID').head(1).ix[:,:7].reset_index(drop = 1).merge(self.VTRACK.groupby('FID').DT.sum().reset_index(), left_on='FID', right_on='FID', how='left')
        
        LabelData = LabelData.merge(ClusterIDdata,left_on = 'FID',right_on = 'FID').merge(self.EnEff,left_on = 'FID',right_on = 'FID',how = 'left')
        
        # Number of clusters in labels, ignoring noise if present.
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        print('Estimated number of clusters: %d' % n_clusters_)
        
        # PLOTTING
        if PLOT == True:
            Median_ID, fig = self.plot_clusters(labels, QuickDraw, MEDIAN, LabelData, verbose)
        
        LabelData['YEAR'] = LabelData.Elap_Time.apply(lambda x: int(x[:4]))
        LabelData['MONTH'] = LabelData.Elap_Time.apply(lambda x: int(x[5:7]))
        LabelData['DAY'] = LabelData.Elap_Time.apply(lambda x: int(x[8:10]))
        LabelData['Hour'] = LabelData.Elap_Time.apply(lambda x: int(x[11:13]))
        if MEDIAN:
            LabelData['MedianID'] =  LabelData.FID.apply(lambda x: Median_ID[x] if x in Median_ID else -99)    # -99 IF NOT MEDIAN
        
        if SAVE == True:
            Fname = 'Label_'+self.Ori+'_'+self.Des+'_'+str(self.year)
            LabelData.to_csv(os.getcwd()+'\TFMS_NEW\\'+Fname+'.csv', index = False)
            pickle.dump(Median_ID, open(os.getcwd()+'\TFMS_NEW\\MEDIAN_'+Fname+'.p','wb'))
        else:
            pass
        if PLOT:
            return LabelData, fig
        else:
            return LabelData, SAVE

    def plot_clusters(self, labels, QuickDraw, MEDIAN, LabelData, verbose):
        Median = []  
        Median_ID = {}
        MeanEff = []
        unique_labels = set(labels)
        boxes = []
        XLegend = []
        colors = ['r','g','m','c','b','y','#ff9933','#660066','#009933','#663300']
        # For paper            
#            fig = plt.figure(figsize=(10.5,3)) #            (18, 8)
        fig = plt.figure(figsize=(18,8)) #            (18, 8)
        ax1 = fig.add_subplot(1,2,1)
#        m = Basemap(llcrnrlon = -126,llcrnrlat = 23.5,urcrnrlon = -65,urcrnrlat = 50,projection='merc')
        m = Basemap(llcrnrlon = self.XLIML,llcrnrlat = self.YLIML,urcrnrlon = self.XLIMU,urcrnrlat = self.YLIMU,projection='merc')
        m.drawmapboundary(fill_color='#8aeaff')
        m.fillcontinents(color='#c5c5c5', lake_color='#8aeaff')

        m.drawcoastlines(linewidth=0.5)
        m.drawcountries(linewidth=0.5)
        m.drawstates(linewidth=0.1)
        
        for k, col in zip(unique_labels, colors):
        
            if k == -1:
                col = 'k'
                ZOrd = 1
                LW = 0.5
                
            else:
                ZOrd = 3
                LW = 0.75
        
            class_member_mask = (labels == k)
            MeanEff.append(self.EnEff[self.EnEff.FID.isin(self.FID_ID[class_member_mask])].Efficiency.mean()*100)
            boxes.append(self.EnEff[self.EnEff.FID.isin(self.FID_ID[class_member_mask])].Efficiency)
#                # Paper
#                Legd = col + ' | col'
            Legd = col + ' | ' + str(sum(class_member_mask)/len(self.FID_ID)*100)[:5] + '%'
            XLegend.append(Legd)
                
            print(k,col,sum(class_member_mask)/len(self.FID_ID), MeanEff[k])
            
            Core_Traj = self.pca.inverse_transform(self.PCA_Comp[class_member_mask]) * self._std + self._mu
            # Center_Traj = self.pca.inverse_transform(cluster_centers)  * self._std + self._mu
            # Center_Traj is not useful here.
            if self.altitude:
                NN = Core_Traj.shape[1]//3
            else:
                NN = Core_Traj.shape[1]//2
            if QuickDraw == True:
                x1, y1 = m(Core_Traj[:,:NN], Core_Traj[:,NN:NN*2])
                ax1.plot(x1,y1, '.', markersize = 1.25, label = 'Core', zorder = ZOrd, c = col)
                
            else:
                for ctraj in Core_Traj:
                    x1,y1 = m(ctraj[:NN],ctraj[NN:2*NN])                
                    ax1.plot(x1,y1,'-',linewidth = LW, label = 'Core',zorder = ZOrd, c=col, alpha = 0.5)
            
            if k != -1:
                Med_PCA, Med_ID = self.NominalRoute(self.PCA_Comp[class_member_mask])
                Median.append(Med_PCA)
                Median_ID[Med_ID] = k
                Median_Traj = self.pca.inverse_transform(Median[k]) * self._std + self._mu
                x_med, y_med = m(Median_Traj[:NN], Median_Traj[NN:NN*2])
                ax1.plot(x_med,y_med, '-', c = 'w', label = 'Nominal', linewidth = 2, zorder = 12)
                if verbose:
                    print('The average travel time is %.2f hours\n' %(LabelData[LabelData.ClustID == k].DT.mean()/3600))
            
        ax1.set_title('Clustering on PCA mode matrix')
        ax2 = fig.add_subplot(1,2,2)
        plt.hold = True
        ax2.boxplot(boxes,vert=1)
        ax2.set_xlabel('Cluster ID | Color', fontsize=8) # 20
        ax2.set_ylabel('En Route Inefficiency', fontsize=8)
        ax2.set_xticklabels(XLegend)
        ax2.set_title('Boxplot of Enroute Inefficiency for Different Clusters', fontsize=8)

        if self.altitude:
            fig, axs = plt.subplots(2, 3, figsize=(18,8), facecolor='w', edgecolor='k')
            axs = axs.ravel()
            for k, col in zip(unique_labels, colors):
                class_member_mask = (labels == k)
                Core_Traj = self.pca.inverse_transform(self.PCA_Comp[class_member_mask]) * self._std + self._mu
                axs[k].set_ylim(0, 450)
                if k != -1:
                    Median_Traj = self.pca.inverse_transform(Median[k]) * self._std + self._mu
                    axs[k].plot(range(NN),Median_Traj[2*NN:3*NN], '-', c = 'w', label = 'Nominal', linewidth = 2., zorder = 12)
                if k == -1:
                    col = 'k'
                    k = max(unique_labels)+1
                for ctraj in Core_Traj:
                    axs[k].plot(range(NN),ctraj[2*NN:3*NN],'-',linewidth = 1.5, label = 'Core',zorder = ZOrd, c=col, alpha = 0.5)

        return Median_ID, fig