# -*- coding: utf-8 -*-
from __future__ import division
import pandas as pd
import numpy as np
import csv
import os
from dateutil.parser import parse
from dateutil.tz import gettz
import time
"""
Created on Fri Aug 05 10:33:29 2016

@ Author: Liu, Yulin
@ Institute: UC Berkeley
"""
"""
This is the file to preprocess the NOAA weather data;
First to decode raw NOAA data by extracting the binary variables;
Then summarize the ARTCC level weather data;
Finally convert the local time to UTC.

"""


# Preprocessing NOAA data
for File in os.listdir(os.getcwd()+'/NOAA/raw'):
    if 'hourly' in File:
        print File
        with open(os.getcwd()+'/NOAA/raw/'+File, 'rb') as csvfile:
            line = csv.reader(csvfile)
            with open(os.getcwd()+'/NOAA/'+File[:12]+'.csv', 'wb') as wcsvfile:
                wline = csv.writer(wcsvfile)
                for row in line:
                    try:
                        wline.writerow([int(row[0]),row[1],row[2],row[8]])
                    except:
                        wline.writerow([row[0],row[1],row[2],row[8]])

st = time.time()                    
StationZone = pd.read_csv(os.getcwd()+'/NOAA/WeatherStation.csv', header = 0, usecols = [0,3,4],dtype={'WBAN':np.int32})
i = 0
for Files in os.listdir(os.getcwd()+'/NOAA/'):
    if 'hourly.csv' in Files:
        if ('2013' in Files) or ('2012' in Files):
            print Files
            i += 1
            if i == 1:
                CurWeather = pd.read_csv(os.getcwd()+'/NOAA/' + Files, header = 0, 
                                     names = ['WBAN','LocalDate','LocalTime','Weather'], 
                                      dtype={'WBAN':np.int32,'LocalDate':np.str,'LocalTime':np.str})
                CurWeather['LocalHour'] = CurWeather['LocalTime'].apply(lambda x: x[:2])
                CurWeather['TS'] = CurWeather['Weather'].apply(lambda x: int('TS' in x))
                CurWeather['TSlevel'] = CurWeather['Weather'].apply(lambda x: int('TS' in x)*2+int('+TS' in x)-int('-TS' in x))
                CurWeather['Hail'] = CurWeather['Weather'].apply(lambda x: int(('GS' in x) or ('GR' in x)))
                CurWeather['Precipitation'] = CurWeather['Weather'].apply(lambda x: int('UP' in x))
                CurWeather['Rain'] = CurWeather['Weather'].apply(lambda x: int('RA' in x))
                CurWeather['Shower'] = CurWeather['Weather'].apply(lambda x: int('SH' in x))
                CurWeather['Ice'] = CurWeather['Weather'].apply(lambda x: int(('PL' in x) or ('IC' in x)))
                CurWeather['Squall'] = CurWeather['Weather'].apply(lambda x: int('SQ' in x))
                FullDataframe = CurWeather
            else:
                print time.time() - st
                CurWeather = pd.read_csv(os.getcwd()+'/NOAA/' + Files, header = 0, 
                                     names = ['WBAN','LocalDate','LocalTime','Weather'], 
                                      dtype={'WBAN':np.int32,'LocalDate':np.str,'LocalTime':np.str})
                CurWeather['LocalHour'] = CurWeather['LocalTime'].apply(lambda x: x[:2])
                CurWeather['TS'] = CurWeather['Weather'].apply(lambda x: int('TS' in x))
                CurWeather['TSlevel'] = CurWeather['Weather'].apply(lambda x: int('TS' in x)*2+int('+TS' in x)-int('-TS' in x))
                CurWeather['Hail'] = CurWeather['Weather'].apply(lambda x: int(('GS' in x) or ('GR' in x)))
                CurWeather['Precipitation'] = CurWeather['Weather'].apply(lambda x: int('UP' in x))
                CurWeather['Rain'] = CurWeather['Weather'].apply(lambda x: int('RA' in x))
                CurWeather['Shower'] = CurWeather['Weather'].apply(lambda x: int('SH' in x))
                CurWeather['Ice'] = CurWeather['Weather'].apply(lambda x: int(('PL' in x) or ('IC' in x)))
                CurWeather['Squall'] = CurWeather['Weather'].apply(lambda x: int('SQ' in x))
                
                FullDataframe = FullDataframe.append(CurWeather)
        else:
            pass
    else:
        pass
FullDataframe = pd.merge(FullDataframe, StationZone, on = 'WBAN', how = 'inner').reset_index(drop = 1)

ZoneSum = FullDataframe.groupby(['ArtZone','LocalDate','LocalHour','TimeZone'])[['TS','TSlevel','Hail','Precipitation',
                                                  'Rain','Shower','Ice','Squall']].agg(['sum','mean'])
                                                  
                                                  
ZoneSum = ZoneSum.reset_index(drop = 0)
ZoneSum.columns = ZoneSum.columns.droplevel()
ZoneSum.columns=['ARTCC','LocalDate','LocalHour','TimeZone','TS_sum','TS_mean','TSlevel_sum','TSlevel_mean','Hail_sum','Hail_mean',
                       'Precipitation_sum','Precipitation_mean','Rain_sum','Rain_mean','Shower_sum','Shower_mean',
                       'Ice_sum','Ice_mean','Squall_sum','Squall_mean']
ZoneSum['LocalTime'] = (ZoneSum['LocalDate'].map(str) + ' ' + ZoneSum['LocalHour'].map(str) + '00').apply(lambda x: parse(x))
ZoneSum.to_csv(os.getcwd()+'/NOAA/ARTCC_Based_Weather_Sum.csv',index = False)

with open(os.getcwd()+'/NOAA/ARTCC_Based_Weather_Sum.csv') as csvfile:
    with open(os.getcwd()+'/NOAA/ARTCC_Based_Weather_Sum_UTC.csv','wb') as wcsvfile:
        rline = csv.reader(csvfile)
        wline = csv.writer(wcsvfile)
        i = 0
        for row in rline:
            i += 1
            if i == 1:
                output = row[:3]
                output.extend(row[4:-1])
                output.extend(['UTCYear','UTCMonth','UTCDay','UTChour'])
                wline.writerow(output)
            else:
                if i % 100000 == 0:
                    print i
                LocalTime = parse(row[-1] + ' LC', tzinfos = {'LC': gettz(row[3])})
                UTCTime = LocalTime.astimezone(gettz('UTC'))
                UTCYear = UTCTime.year
                UTCMonth = UTCTime.month
                UTCDay = UTCTime.day
                UTCHour = UTCTime.hour
                output = row[:3]
                output.extend(row[4:-1])
                output.extend([UTCYear,UTCMonth,UTCDay,UTCHour])
                if UTCYear == 2013:
                    wline.writerow(output)
                else:
                    pass
                