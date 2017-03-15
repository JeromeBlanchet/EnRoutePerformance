# -*- coding: utf-8 -*-
"""
Created on Tue Jul 05 10:23:03 2016

@ Author: Liu, Yulin
@ Institute: UC Berkeley
"""
"""
This file:
 - combines the monthly hourly binary weather data into one fil;
 - converts the local time to UTC time;
 - decode the strings to binary variables

"""


from __future__ import division
import pandas as pd
import numpy as np
import csv
import os
from dateutil.parser import parse
from dateutil.tz import gettz
import geopandas as gpd
import shapely
import time
from datetime import datetime

def ConvertTime(finname,foutname):
    ToTZ = gettz('UTC')
    with open(finname,'r') as csvfile:
        line = csv.reader(csvfile)
        next(line)
        with open(foutname,'wb') as wcsvfile:
            wline = csv.writer(wcsvfile)            
            for row in line:
                if row[1][:4] == '2014':
                    pass
                else:
                    if len(row[2]) == 1:
                        row[2] = '000' + row[2]
                    elif len(row[2]) == 2:
                        row[2] = '00' + row[2]
                    elif len(row[2]) == 3:
                        row[2] = '0' + row[2]
                    else:
                        pass
                    FromTZ = parse(row[1] + ' ' + row[2] +' LC',tzinfos = {'LC':gettz(row[4])})
                    ToZone = str(FromTZ.astimezone(ToTZ))[:19]
                    if ToZone[:4] == '2013':
                        wline.writerow([row[0],row[2],row[5],ToZone[:4],ToZone[5:7],ToZone[8:10],ToZone[11:13],ToZone[14:16]])
                    else:
                        pass

                   
def ExtractWeather(Station,WeatherYear = 2013):
    ToTimeZone = gettz('UTC')
    with open(os.getcwd()+'/NOAA/'+str(WeatherYear) + 'Weather.csv','wb') as wcsvfile:
        wline = csv.writer(wcsvfile)
        Header = ['WBAN','LocalDate','LocalTime','ARTCC','UTCmonth','UTCday','UTChour','UTCmin',
                  'TS','TSlevel','Hail','Precipitation','Rain','Shower','Ice','Squall']
        wline.writerow(Header)
        for Files in os.listdir(os.getcwd()+'/NOAA/'):
            if 'hourly.csv' in Files:
                if (str(WeatherYear) in Files) or (str(WeatherYear-1) in Files):
                    Fname = Files
                    with open(os.getcwd()+'/NOAA/' + Fname,'r') as rcsvfile:
                        print(Fname)
                        print(datetime.now())
                        rline = csv.reader(rcsvfile,delimiter=',')
                        next(rline)
                        for row in rline:
                            try:
                                ARTCC = Station[row[0]][0]
                                FromTimeZone = Station[row[0]][1]
                                if len(row[2]) == 1:
                                    row[2] = '000' + row[2]
                                elif len(row[2]) == 2:
                                    row[2] = '00' + row[2]
                                elif len(row[2]) == 3:
                                    row[2] = '0' + row[2]
                                else:
                                    pass
                                
                                ThunderStorm = int('TS' in row[3])
                                ThunderStormLevel = int('TS' in row[3]) * 2 + int('+TS' in row[3]) - int('-TS' in row[3])
                                Hail = int(('GR' in row[3]) or ('GS' in row[3]))
                                Precipitation = int('UP' in row[3])
                                Rain = int('RA' in row[3])
                                Ice = int(('PL' in row[3]) or ('IC' in row[3]))
                                Squall = int('SQ' in row[3])
                                Shower = int('SH' in row[3])
               
                                LocalTime = parse(row[1] + ' ' + row[2] +' LC',tzinfos = {'LC':gettz(FromTimeZone)})
                                UTCTime = str(LocalTime.astimezone(ToTimeZone))[:19]
                                if UTCTime[:4] == '2013':
                                    OutputLine = [row[0],row[1],row[2],ARTCC,UTCTime[5:7],UTCTime[8:10],UTCTime[11:13],UTCTime[14:16],
                                              ThunderStorm, ThunderStormLevel, Hail, Precipitation, Rain, Shower, Ice, Squall]
                                    wline.writerow(OutputLine)
                                else:
                                    pass
                            except KeyError:
                                pass
                else:
                    pass
            else:
                pass
            

def ExtractWeather2(WeatherYear = 2013):
    StationZone = pd.read_csv(os.getcwd()+'/NOAA/WeatherStation.csv', header = 0, usecols = [0,3,4])
#    Header = ['WBAN','LocalDate','LocalTime','ARTCC','UTCmonth','UTCday','UTChour','UTCmin',
#                  'TS','TSlevel','Hail','Precipitation','Rain','Shower','Ice','Squall']
    FullDataframe = pd.DataFrame({'WBAN':[], 'LocalDate':[], 'LocalTime':[], 'Weather':[]})
    for Files in os.listdir(os.getcwd()+'/NOAA/'):
        if 'hourly.csv' in Files:
            if ('2013' in Files) or ('2012' in Files):
                CurWeather = pd.read_csv(os.getcwd()+'/NOAA/' + Files, header = 0, 
                                     names = ['WBAN','LocalDate','LocalTime','Weather'], 
                                      dtype={'WBAN':'int','LocalDate':'int','LocalTime':'int'})
                FullDataframe = FullDataframe.append(CurWeather)
            else:
                pass
        else:
            pass
    FullDataframe = pd.merge(FullDataframe, StationZone, on = 'WBAN', how = 'inner').reset_index(drop = 1)
    
    
Station = {}
with open(os.getcwd()+'\NOAA\\WeatherStation.csv','r') as csvfile:
    line = csv.reader(csvfile)
    next(line)
    for row in line:
        Station[row[0]] = [row[3],row[4]]                   

ExtractWeather(Station)


#FinName = ['ZHU.csv','ZFW.csv','ZME.csv','ZID.csv','ZDC.csv','ZNY.csv','ZBW.csv','ZAB.csv',
#           'ZAU.csv','ZDV.csv','ZJX.csv','ZOA.csv','ZKC.csv','ZLA.csv','ZLC.csv','ZMP.csv',
#           'ZSE.csv','ZTL.csv','ZMA.csv','ZOB.csv']
#           
#for name in FinName:
#    print(name)
#    Fin = os.getcwd()+'\NOAA\\'+name
#    Fout = os.getcwd()+'\NOAA\\New_'+name
#    ConvertTime(Fin, Fout)
        
        
        
    
    
