# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 09:51:38 2016

@ Author: Liu, Yulin
@ Institute: UC Berkeley
"""

from __future__ import division
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


class Traj_Clustering:
    def __init__(self,Ori, Des, year, N_pt = 100, N_Comp = 4, VTRACK = None, EnEff = None,Prefer = False):
        self.Ori = Ori
        self.Des = Des
        self.year = year
        self.Coords = GetCoords()
        self.Ori_Lat = self.Coords[self.Ori][0]
        self.Ori_Lon = self.Coords[self.Ori][1]
        self.Des_Lat = self.Coords[self.Des][0]
        self.Des_Lon = self.Coords[self.Des][1]
        if VTRACK is None:
            try:
                self.VTRACK = pd.read_csv(os.getcwd()+'\TFMS_NEW\\'+'New_'+self.Ori+self.Des+str(self.year)+'.csv')
            except:
                raise ValueError('No trajectory data found! Please load valid flight track data!')
#                DATA = EDA_Data(self.Ori,self.Des,self.year)
#                self.VTRACK = DATA.VTrack
        else:
            self.VTRACK = VTRACK.copy()
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
        
        self.N_pt = N_pt
        self.N_Comp = N_Comp
        
        self.NewSeries = self.Interpolation()
        self.PCA_Inv, self.PCA_Comp, self.pca = self.PCA_Traj()
        
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
            OldSeries.append(np.append(xold,yold))
            NewSeries.append(np.append(xnew,ynew))
            
        NewSeries = np.asarray(NewSeries)
        OldSeries = np.asarray(OldSeries)
        return NewSeries
        
    def PCA_Traj(self):
        
        pca = PCA(n_components = self.N_Comp)
        PCA_Compon = pca.fit_transform(self.NewSeries)
        print(pca.explained_variance_ratio_)
        print( sum(pca.explained_variance_ratio_) )
        PCA_Inv = pca.inverse_transform(PCA_Compon)
        return PCA_Inv,PCA_Compon,pca
    
    def Plot_PCA_Int(self,xlb = -126,xrb = -80, ylb = 26,yub = 50):
        self.XLIMU = xrb
        self.XLIML = xlb
        self.YLIMU = yub
        self.YLIML = ylb
        N_pnt = self.NewSeries.shape[1]//2
        
        fig = plt.figure(figsize=(16,6))
        ax1 = fig.add_subplot(121)
        ax1.scatter(self.NewSeries[:,:N_pnt],self.NewSeries[:,N_pnt:],edgecolor = 'b', s = 0.01, label = 'Interpolated')
        ax2 = fig.add_subplot(122,sharex=ax1,sharey=ax1)
        ax2.scatter(self.PCA_Inv[:,:N_pnt],self.PCA_Inv[:,N_pnt:],edgecolor = 'g',s = 0.01, label = 'Inverse PCA')
        
        ax1.set_xlim(self.XLIML,self.XLIMU)
        ax1.set_ylim(self.YLIML,self.YLIMU)
        ax1.set_title('Original Resampled Trajectories')
        ax1.set_xlabel('Lon')
        ax1.set_ylabel('Lat')
        ax2.set_title('Trajectories based on PCA with five factors')
        ax2.set_xlabel('Lon')
        ax2.set_ylabel('Lat')
    
    def Dist_Mat(self,x):
        DMat = squareform(pdist(x,'euclidean'))
        return DMat
    
    def NominalRoute(self,x):
        Dist_matrix = self.Dist_Mat(x)
        TDist = sum(Dist_matrix)
        Idx = np.argmin(TDist)
        return x[Idx], self.FID_ID[np.where(self.PCA_Comp == x[Idx])[0][0]]
    
    def DB_Clustering(self, EPS, Min_Samp, PLOT = True, MEDIAN = False, Prefer = False, SAVE = False, QuickDraw = True):
        dbscan = cluster.DBSCAN(eps = EPS, min_samples = Min_Samp)
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
            m.bluemarble()
            m.drawcoastlines(linewidth=0.5)
            m.drawcountries(linewidth=0.5)
            m.drawstates(linewidth=0.5)
            m.drawparallels(np.arange(10.,35.,5.))
            m.drawmeridians(np.arange(-120.,-80.,10.))
            m.drawgreatcircle(self.Ori_Lon,self.Ori_Lat,self.Des_Lon,self.Des_Lat,linewidth=3,color='w',linestyle='--',zorder = 10)
            
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
                    
                print( k,col,sum(class_member_mask)/len(self.FID_ID), MeanEff[k] )
                
                Core_Traj = self.pca.inverse_transform(self.PCA_Comp[class_member_mask & core_samples_mask])
                Member_Traj = self.pca.inverse_transform(self.PCA_Comp[class_member_mask & ~core_samples_mask]) 
                
    #            Core_Traj = self.NewSeries[class_member_mask & core_samples_mask]
    #            Member_Traj = self.NewSeries[class_member_mask & ~core_samples_mask]
                NN = Core_Traj.shape[1]                
                if QuickDraw == True:
                    x1, y1 = m(Core_Traj[:,:int(NN/2)], Core_Traj[:,int(NN/2):])
                    ax1.plot(x1,y1, '.', markersize = 1.25, label = 'Core', zorder = ZOrd, c = col)
                    
                    x2,y2 = m(Member_Traj[:,:int(NN/2)], Member_Traj[:,int(NN/2):])                
                    ax1.plot(x2,y2, '.', markersize = 1, label = 'Core', zorder = ZOrd, c = col)
                else:
                    for ctraj in Core_Traj:
                        x1,y1 = m(ctraj[:int(NN/2)],ctraj[int(NN/2):])                
                        ax1.plot(x1,y1,'-',linewidth = LW, label = 'Core',zorder = ZOrd, c=col)
                    for mtraj in Member_Traj:
                        x2,y2 = m(mtraj[:int(NN/2)],mtraj[int(NN/2):])                
                        ax1.plot(x2,y2,'-',linewidth = LW,label = 'Core',zorder = ZOrd, c=col)
    
    #            Longi1 = Core_Traj[:,:int(NN/2)]
    #            Latit1 = Core_Traj[:,int(NN/2):]
    #            Longi2 = Member_Traj[:,:int(NN/2)]
    #            Latit2 = Member_Traj[:,int(NN/2):]
    #            x1, y1 = m(Longi1.reshape(-1,1), Latit1.reshape(-1,1))
    #            x2, y2 = m(Longi2.reshape(-1,1), Latit2.reshape(-1,1))
    #            ax1.scatter(x1,y1,edgecolors = 'face',label = 'Core',c = col, s = 0.3,zorder = ZOrd)            
    #            ax1.scatter(x2,y2,edgecolors = 'face',label = 'Mem',c = col, s = 0.3,zorder = ZOrd)
                
                if MEDIAN == True:
                    if k != -1:
                        Med_PCA, Med_ID = self.NominalRoute(self.PCA_Comp[class_member_mask])
                        Median.append(Med_PCA)
                        Median_ID[Med_ID] = k
                        Median_Traj = self.pca.inverse_transform(Median[k])
                        x_med, y_med = m(Median_Traj[:int(NN/2)], Median_Traj[int(NN/2):])
                        ax1.plot(x_med,y_med, '-', c = 'w', label = 'Nominal', linewidth = 2, zorder = 12)
#                        print('The Median FID is: %d' %Med_ID)
                        print('The average travel time is %.2f hours\n' %(LabelData[LabelData.ClustID == k].DT.mean()/3600))
                else:
                    pass
    #            
    #            if Prefer == True:
    #                for i in range(len(self.PrefRoute)):
    #                    ax1.plot(self.PrefRoute[str(i)][:,1],self.PrefRoute[str(i)][:,0],'ok--', zorder = 20,lw = 1)
    #                for i in range(len(self.CDRoute)):
    #                    ax1.plot(self.CDRoute[str(i)][:,1],self.CDRoute[str(i)][:,0],'o--', c = '#e6e600',zorder = 15,lw = 2)
    #            else:
    #                pass
                
            ax1.set_title('DBSCAN applied to PCA mode matrix')
            ax2 = fig.add_subplot(1,2,2)
            plt.hold = True
            ax2.boxplot(boxes,vert=1)
            ax2.set_xlabel('Cluster ID | Color', fontsize=8) # 20
            ax2.set_ylabel('En Route Inefficiency', fontsize=8)
            ax2.set_xticklabels(XLegend)
            ax2.set_title('Boxplot of Enroute Inefficiency for Different Clusters', fontsize=8)
        
        LabelData['YEAR'] = LabelData.Elap_Time.apply(lambda x: int(x[:4]))
        LabelData['MONTH'] = LabelData.Elap_Time.apply(lambda x: int(x[5:7]))
        LabelData['DAY'] = LabelData.Elap_Time.apply(lambda x: int(x[8:10]))
        LabelData['Hour'] = LabelData.Elap_Time.apply(lambda x: int(x[11:13]))
        if MEDIAN:
            LabelData['MedianID'] =  LabelData.FID.apply(lambda x: Median_ID[x] if Median_ID.has_key(x) else -2)    # -2 IF NOT MEDIAN
        
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
            
    def Kmeans_Clustering(self, n_clust, PLOT = True, MEDIAN = False, Prefer = False, SAVE = False, QuickDraw = True):
#        clustering_names = [
#    'MiniBatchKMeans', 'AffinityPropagation', 'MeanShift',
#    'SpectralClustering', 'Ward', 'AgglomerativeClustering',
#    'DBSCAN', 'Birch']
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
            m.bluemarble()
            m.drawcoastlines(linewidth=0.5)
            m.drawcountries(linewidth=0.5)
            m.drawstates(linewidth=0.5)
            m.drawparallels(np.arange(10.,35.,5.))
            m.drawmeridians(np.arange(-120.,-80.,10.))
            m.drawgreatcircle(self.Ori_Lon,self.Ori_Lat,self.Des_Lon,self.Des_Lat,linewidth=3,color='w',linestyle='--',zorder = 10)
            
            for k, col in zip(unique_labels, colors):
            
                class_member_mask = (labels == k)
                MeanEff.append(self.EnEff[self.EnEff.FID.isin(self.FID_ID[class_member_mask])].Efficiency.mean()*100)
                boxes.append(self.EnEff[self.EnEff.FID.isin(self.FID_ID[class_member_mask])].Efficiency)
#                # Paper
#                Legd = col + ' | col'
                Legd = col + ' | ' + str(sum(class_member_mask)/len(self.FID_ID)*100)[:5] + '%'
                XLegend.append(Legd)
                    
                print( k,col,sum(class_member_mask)/len(self.FID_ID), MeanEff[k] )
                
                Core_Traj = self.pca.inverse_transform(self.PCA_Comp[class_member_mask])
                
    #            Core_Traj = self.NewSeries[class_member_mask & core_samples_mask]
    #            Member_Traj = self.NewSeries[class_member_mask & ~core_samples_mask]
                NN = Core_Traj.shape[1]                
                if QuickDraw == True:
                    x1, y1 = m(Core_Traj[:,:int(NN/2)], Core_Traj[:,int(NN/2):])
                    ax1.plot(x1,y1, '.', markersize = 1.25, c = col)
                    
                else:
                    for ctraj in Core_Traj:
                        x1,y1 = m(ctraj[:int(NN/2)],ctraj[int(NN/2):])                
                        ax1.plot(x1,y1,'-',linewidth = 0.5, c=col)
    
                
                if MEDIAN == True:
                    if k != -1:
                        Med_PCA, Med_ID = self.NominalRoute(self.PCA_Comp[class_member_mask])
                        Median.append(Med_PCA)
                        Median_ID[Med_ID] = k
                        Median_Traj = self.pca.inverse_transform(Median[k])
                        x_med, y_med = m(Median_Traj[:int(NN/2)], Median_Traj[int(NN/2):])
                        ax1.plot(x_med,y_med, '-', c = 'w', label = 'Nominal', linewidth = 2, zorder = 12)
#                        print('The Median FID is: %d' %Med_ID)
                        print('The average travel time is %.2f hours\n' %(LabelData[LabelData.ClustID == k].DT.mean()/3600))
                else:
                    pass
                
            ax1.set_title('k-means applied to PCA mode matrix')
            ax2 = fig.add_subplot(1,2,2)
            plt.hold = True
            ax2.boxplot(boxes,vert=1)
            ax2.set_xlabel('Cluster ID | Color', fontsize=8) # 20
            ax2.set_ylabel('En Route Inefficiency', fontsize=8)
            ax2.set_xticklabels(XLegend)
            ax2.set_title('Boxplot of Enroute Inefficiency for Different Clusters', fontsize=8)
        
        LabelData['YEAR'] = LabelData.Elap_Time.apply(lambda x: int(x[:4]))
        LabelData['MONTH'] = LabelData.Elap_Time.apply(lambda x: int(x[5:7]))
        LabelData['DAY'] = LabelData.Elap_Time.apply(lambda x: int(x[8:10]))
        LabelData['Hour'] = LabelData.Elap_Time.apply(lambda x: int(x[11:13]))
        if MEDIAN:
            LabelData['MedianID'] =  LabelData.FID.apply(lambda x: Median_ID[x] if Median_ID.has_key(x) else -2)    # -2 IF NOT MEDIAN
        
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
            m.bluemarble()
            m.drawcoastlines(linewidth=0.5)
            m.drawcountries(linewidth=0.5)
            m.drawstates(linewidth=0.5)
            m.drawparallels(np.arange(10.,35.,5.))
            m.drawmeridians(np.arange(-120.,-80.,10.))
            m.drawgreatcircle(self.Ori_Lon,self.Ori_Lat,self.Des_Lon,self.Des_Lat,linewidth=3,color='w',linestyle='--',zorder = 10)
            
            for k, col in zip(unique_labels, colors):
            
                class_member_mask = (labels == k)
                MeanEff.append(self.EnEff[self.EnEff.FID.isin(self.FID_ID[class_member_mask])].Efficiency.mean()*100)
                boxes.append(self.EnEff[self.EnEff.FID.isin(self.FID_ID[class_member_mask])].Efficiency)
#                # Paper
#                Legd = col + ' | col'
                Legd = col + ' | ' + str(sum(class_member_mask)/len(self.FID_ID)*100)[:5] + '%'
                XLegend.append(Legd)
                    
                print( k,col,sum(class_member_mask)/len(self.FID_ID), MeanEff[k] )
                
                Core_Traj = self.pca.inverse_transform(self.PCA_Comp[class_member_mask])
                
    #            Core_Traj = self.NewSeries[class_member_mask & core_samples_mask]
    #            Member_Traj = self.NewSeries[class_member_mask & ~core_samples_mask]
                NN = Core_Traj.shape[1]                
                if QuickDraw == True:
                    x1, y1 = m(Core_Traj[:,:int(NN/2)], Core_Traj[:,int(NN/2):])
                    ax1.plot(x1,y1, '.', markersize = 1.25, c = col)
                    
                else:
                    for ctraj in Core_Traj:
                        x1,y1 = m(ctraj[:int(NN/2)],ctraj[int(NN/2):])                
                        ax1.plot(x1,y1,'-',linewidth = 0.5, c=col)
    
                
                if MEDIAN == True:
                    if k != -1:
                        Med_PCA, Med_ID = self.NominalRoute(self.PCA_Comp[class_member_mask])
                        Median.append(Med_PCA)
                        Median_ID[Med_ID] = k
                        Median_Traj = self.pca.inverse_transform(Median[k])
                        x_med, y_med = m(Median_Traj[:int(NN/2)], Median_Traj[int(NN/2):])
                        ax1.plot(x_med,y_med, '-', c = 'w', label = 'Nominal', linewidth = 2, zorder = 12)
#                        print('The Median FID is: %d' %Med_ID)
                        print('The average travel time is %.2f hours\n' %(LabelData[LabelData.ClustID == k].DT.mean()/3600))
                else:
                    pass
                
            ax1.set_title('Spectral Clustering applied to PCA mode matrix')
            ax2 = fig.add_subplot(1,2,2)
            plt.hold = True
            ax2.boxplot(boxes,vert=1)
            ax2.set_xlabel('Cluster ID | Color', fontsize=8) # 20
            ax2.set_ylabel('En Route Inefficiency', fontsize=8)
            ax2.set_xticklabels(XLegend)
            ax2.set_title('Boxplot of Enroute Inefficiency for Different Clusters', fontsize=8)
        
        LabelData['YEAR'] = LabelData.Elap_Time.apply(lambda x: int(x[:4]))
        LabelData['MONTH'] = LabelData.Elap_Time.apply(lambda x: int(x[5:7]))
        LabelData['DAY'] = LabelData.Elap_Time.apply(lambda x: int(x[8:10]))
        LabelData['Hour'] = LabelData.Elap_Time.apply(lambda x: int(x[11:13]))
        if MEDIAN:
            LabelData['MedianID'] =  LabelData.FID.apply(lambda x: Median_ID[x] if Median_ID.has_key(x) else -2)    # -2 IF NOT MEDIAN
        
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
            m.bluemarble()
            m.drawcoastlines(linewidth=0.5)
            m.drawcountries(linewidth=0.5)
            m.drawstates(linewidth=0.5)
            m.drawparallels(np.arange(10.,35.,5.))
            m.drawmeridians(np.arange(-120.,-80.,10.))
            m.drawgreatcircle(self.Ori_Lon,self.Ori_Lat,self.Des_Lon,self.Des_Lat,linewidth=3,color='w',linestyle='--',zorder = 10)
            
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
                    
                print( k,col,sum(class_member_mask)/len(self.FID_ID), MeanEff[k] ) 
                
                Core_Traj = self.pca.inverse_transform(self.PCA_Comp[class_member_mask])
                Center_Traj = self.pca.inverse_transform(cluster_centers) 
                # Center_Traj is not useful here.
                
    #            Core_Traj = self.NewSeries[class_member_mask & core_samples_mask]
    #            Member_Traj = self.NewSeries[class_member_mask & ~core_samples_mask]
                NN = Core_Traj.shape[1]                
                if QuickDraw == True:
                    x1, y1 = m(Core_Traj[:,:int(NN/2)], Core_Traj[:,int(NN/2):])
                    ax1.plot(x1,y1, '.', markersize = 1.25, label = 'Core', zorder = ZOrd, c = col)
                    
                else:
                    for ctraj in Core_Traj:
                        x1,y1 = m(ctraj[:int(NN/2)],ctraj[int(NN/2):])                
                        ax1.plot(x1,y1,'-',linewidth = LW, label = 'Core',zorder = ZOrd, c=col)
                
                if MEDIAN == True:
                    if k != -1:
                        Med_PCA, Med_ID = self.NominalRoute(self.PCA_Comp[class_member_mask])
                        Median.append(Med_PCA)
                        Median_ID[Med_ID] = k
                        Median_Traj = self.pca.inverse_transform(Median[k])
                        x_med, y_med = m(Median_Traj[:int(NN/2)], Median_Traj[int(NN/2):])
                        ax1.plot(x_med,y_med, '-', c = 'w', label = 'Nominal', linewidth = 2, zorder = 12)
#                        print('The Median FID is: %d' %Med_ID)
                        print('The average travel time is %.2f hours\n' %(LabelData[LabelData.ClustID == k].DT.mean()/3600))
                else:
                    pass
    #            
    #            if Prefer == True:
    #                for i in range(len(self.PrefRoute)):
    #                    ax1.plot(self.PrefRoute[str(i)][:,1],self.PrefRoute[str(i)][:,0],'ok--', zorder = 20,lw = 1)
    #                for i in range(len(self.CDRoute)):
    #                    ax1.plot(self.CDRoute[str(i)][:,1],self.CDRoute[str(i)][:,0],'o--', c = '#e6e600',zorder = 15,lw = 2)
    #            else:
    #                pass
                
            ax1.set_title('Mean-shift applied to PCA mode matrix')
            ax2 = fig.add_subplot(1,2,2)
            plt.hold = True
            ax2.boxplot(boxes,vert=1)
            ax2.set_xlabel('Cluster ID | Color', fontsize=8) # 20
            ax2.set_ylabel('En Route Inefficiency', fontsize=8)
            ax2.set_xticklabels(XLegend)
            ax2.set_title('Boxplot of Enroute Inefficiency for Different Clusters', fontsize=8)
        
        LabelData['YEAR'] = LabelData.Elap_Time.apply(lambda x: int(x[:4]))
        LabelData['MONTH'] = LabelData.Elap_Time.apply(lambda x: int(x[5:7]))
        LabelData['DAY'] = LabelData.Elap_Time.apply(lambda x: int(x[8:10]))
        LabelData['Hour'] = LabelData.Elap_Time.apply(lambda x: int(x[11:13]))
        if MEDIAN:
            LabelData['MedianID'] =  LabelData.FID.apply(lambda x: Median_ID[x] if Median_ID.has_key(x) else -2)    # -2 IF NOT MEDIAN
        
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
