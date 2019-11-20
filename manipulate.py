def warn(*args, **kwargs):
    pass
import warnings

import seaborn as sns
import matplotlib.pyplot as plt
from prince import MCA
warnings.warn = warn
from sklearn.exceptions import DataConversionWarning
import pandas as pd
from datetime import datetime as dt
import numpy as np
import math
from forecastweather import Temp, Wind, Humid, Weather

train = pd.read_csv('data_train.csv')
test = pd.read_csv('data_test.csv')
sheet = pd.read_csv('Sheet.csv')

train_row = train.shape[0]
df = pd.concat([train, test], axis = 0, sort = True).reset_index(drop = True)

df['ex'] = df.x_entry.apply(lambda x: 2 if 3750901.5068 <= x <= 3770901.5068 else 1 if  ((3747901.5068 < x < 3750901.5068) or (3770901.5068 < x < 3773901.5068)) else -0.5)
df['ex_1'] = df.x_entry.apply(lambda x: 4 if 3755901.507 <= x <= 3765901.507 else 0)
df['ey'] = df.y_entry.apply(lambda x: 2 if -19268905.6133 <= x <= -19208905.6133 else 1 if ((-19298905.6133 < x < -19268905.6133) or (-19208905.6133 < x < -19178905.6133)) else -0.5)
df['ey_1'] = df.y_entry.apply(lambda x: 4 if -19253905.61 <= x <= -19223905.61 else 0)
df['ez'] = df['ex'] + df['ey'] + df['ey_1'] + df['ex_1']

def function(z):
    if z == 12:
        return 5
    elif z == 10 or z == 8 or z == 4:
        return 4
    elif z == 7 or z == 3 or z == 2:
        return 3
    else:
        return 2

df['incity_entry'] = df['ez'].apply(lambda x: function(x))
df['incity_entry'] = [1 if i >= -19121358 or i <= -19352500 or k <= 3744000 else j for k, i, j in zip(df.x_entry, df.y_entry, df.incity_entry)]
df['time_entry'] = pd.to_datetime(df['time_entry'], format = '%H:%M:%S')
df['time_exit'] = pd.to_datetime(df['time_exit'], format = '%H:%M:%S')
df['time_gap'] = df['time_exit'] - df['time_entry']
df['time_gap'] = [round(_.total_seconds()/60, 4) + 0.05 for _ in df['time_gap']]
df['entry_hour'] = [float(_.hour) for _ in df['time_entry']]
df['entry_minute'] = [float(_.minute) for _ in df['time_entry']]

df['time_'] = round(df['entry_hour'] + df['entry_minute'], 2)
df['traffic'] = df['time_'].apply(lambda x: (np.sin((x + 3.3) * 0.5) + 1.1))

df['trajectory_n1'] = [int(_.split('_')[2]) for _ in df['trajectory_id']]
df['trajectory_n2'] = [int(_.split('_')[3]) + 1 for _ in df['trajectory_id']]
df['WkDay'] = df.apply(lambda i:dt.strptime('%s %s %s' %(2018, 10, i.trajectory_n1), '%Y %m %d'), axis = 1)
df['WkDay'] = [dt.isoweekday(a) for a in df.WkDay]

df = df.sort_values(['hash', 'trajectory_n1', 'trajectory_n2'])

df['Delimiter'] = [str(i) + '-' + str(j) for i, j in zip(df.trajectory_n1, df.entry_hour)]

df['temp'] = df['Delimiter'].map(Temp)
#df['humid'] = df['Delimiter'].map(Humid)
df['temp'] = df.groupby(['WkDay'])['temp'].apply(lambda x: x.fillna(x.mean()))
#df['humid'] = df.groupby(['WkDay'])['humid'].apply(lambda x: x.fillna(x.mean()))

df['weather'] = df['Delimiter'].map(Weather)
df['weather'] = df.weather.fillna(3)

df['previous_x_exit'] = df.groupby(['hash', 'trajectory_n1'])['x_exit'].shift(1)
df['previous_x_exit'] = df.groupby(['hash', 'trajectory_n1']).transform(df.previous_x_exit.interpolate(method='spline', order = 1, limit_direction = 'backward'))
df['previous_x_exit'] = df.groupby(['incity_entry', 'WkDay', 'entry_hour'])['previous_x_exit'].apply(lambda x: x.fillna(x.mean()))
df['previous_x_exit'] = round(df['previous_x_exit'], 4)

df['previous_y_exit'] = df.groupby(['hash', 'trajectory_n1'])['y_exit'].shift(1)
df['previous_y_exit'] = df.groupby(['hash', 'trajectory_n1']).transform(df.previous_x_exit.interpolate(method='spline', order = 1, limit_direction = 'backward'))
df['previous_y_exit'] = df.groupby(['incity_entry', 'WkDay', 'entry_hour'])['previous_y_exit'].apply(lambda x: x.fillna(x.mean()))
df['previous_y_exit']  = round(df['previous_y_exit'], 4)

def coordinate_angle(x1, y1, x2, y2):
    X = math.degrees(math.atan2(y2 - y1, x2 - x1))
    return X

def euclidean(x2, y2, x1, y1):
    X = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return X

df['center_angle'] = df[['x_entry', 'y_entry']].apply(lambda x : coordinate_angle(*x, 3760901.5068, -19238905.6133), axis = 1)
df['prev_distance'] = df[['previous_x_exit', 'previous_y_exit']].apply(lambda x : np.log2(euclidean(*x, 3760901.5068, -19238905.6133)) + 0.01, axis = 1)
df['distance_gap'] = df[['previous_x_exit', 'previous_y_exit', 'x_entry', 'y_entry']].apply(lambda x : np.log2(euclidean(*x) + 1), axis = 1)
df['pace'] = df['distance_gap'] / df['time_gap']
df['distance_gap'] = df[['x_entry', 'y_entry']].apply(lambda x : np.log2(euclidean(*x, 3760901.5068, -19238905.6133)) + 0.01, axis = 1)
df['distance_gap'] = df['distance_gap'] - df['prev_distance']

df['city_yn'] = df['incity_entry'].apply(lambda x : 1 if x > 2 else 0)
                                     
df['prevexit_citystate'] = df.groupby(['hash', 'trajectory_n1'])['city_yn'].cumsum()
df['prevexit_citystate'] = df['prevexit_citystate'].fillna(0)
df['temp_citystate'] = df.groupby(['hash', 'trajectory_n1'])['prevexit_citystate'].shift(1).fillna(0)
df = pd.merge(df, sheet, how = 'left', on = ['trajectory_id']).rename(columns = {'Value' : 'highway'})
df['highway'] = [euclidean(x, y, 3760901.5068, -19238905.6133) + 100000 if i == 1 else euclidean(x, y, 3760901.5068, -19238905.6133)
                         for x, y, i in zip(df.x_entry, df.y_entry, df.highway)]

def test(x, y):
    Z = 0
    if x == y:
        Z = -1
    return Z

def combine(x, y):
    z = None
    if x == 0:
        z = y
    elif x != 0:
        z = x
    return z

def combine_1(x,y):
    z = None
    if x > 0 and x == y:
        z = 0
    else:
        z = x
    return z

df['temp_citystate'] = df[['prevexit_citystate', 'temp_citystate']].apply(lambda x : test(*x), axis = 1)
df['temp_citystate'] = df.groupby(['hash', 'trajectory_n1', 'prevexit_citystate'])['temp_citystate'].cumsum()
df['prev_state'] = df[['prevexit_citystate', 'temp_citystate']].apply(lambda x : combine(*x), axis = 1)

df['trajectory_n1'] = np.searchsorted([6, 13, 20, 27], df['trajectory_n1']).astype(int)

df['x_entry'] = -((df['x_entry'] - 3750901.5068) * (df['x_entry'] - 3770901.5068)) / 10000
df['y_entry'] = -((df['y_entry'] + 19268905.6133) * (df['y_entry'] + 19208905.6133)) / 100000
df['para_angle'] = df[['x_entry', 'y_entry']].apply(lambda x : coordinate_angle(*x, -10000, -100000), axis = 1)

def Highway(function, data_input, data_target, initial_drange, end_drange, r, max_thres, filter_I, filter_II, crt = [1,2,3,4,5,6], mode = 'Linear', const = 1):
    if initial_drange < data_input <= end_drange:
        A = round(function, 4)
        if mode == 'Linear':
            if (data_target - r) < A < (data_target + r):
                if (filter_I >= 90000 and filter_II >= 10) or\
                   (filter_I >= 80000 and filter_II >= 20) or\
                   (filter_I <= -15000 and filter_II >= 15):
                    return int(2)
                else:
                    return int(1)
                    
            elif ((A - r) > data_target > (A - 1.75*r)) or ((A + r) < data_target < (A + 1.75*r)):
                return int(-1)
                
            elif ((A - 1.75*r) > data_target >= (A - 2.2*r)) or ((A + 1.75*r) < data_target <= (A + 2.2*r)):
                return int(-2)
                
        elif mode == 'Parabola':
            if (initial_drange >= 0 and end_drange >= 0) or (initial_drange < 0 and end_drange < 0):
                 o = initial_drange + (abs(end_drange - initial_drange)/2)
            else:
                o = initial_drange + ((abs(initial_drange) + abs(end_drange))/2)

            i = (-(o - initial_drange) * (o - end_drange))/(max_thres - r)
            r = round((-(data_input - initial_drange) * (data_input - end_drange)/i) + r, 2)
            if (data_target - r) < A < (data_target + r):
                if (filter_I >= crt[0] and filter_II >= crt[3]) or (filter_I >= crt[1] and filter_II >= crt[4]) or (filter_I <= crt[2] and filter_II >= crt[5]):
                    return int(2)
                else:
                    return int(1)
        
            elif ((A - r) > data_target >= (A - const*1.75*r)) or ((A + r) < data_target <= (A + const*1.75*r)):
                return int(-1)

            elif ((A - const*1.75*r) > data_target >= (A - const*2.2*r)) or ((A + const*1.75*r) < data_target <= (A + const*2.2*r)):
                return int(-2)
    return int(-3)

def Quadratic(a,b,c):
    square = (b**2 - 4*a*c)
    if square > 0:
        upper = (- b + (square)**0.5)/2*a
        lower  =(- b - (square)**0.5)/2*a
        return upper, lower
    else:
        return 'Complex', 'Complex'

def RadiusSearch(x, y, radius, h_pos, k_pos, filter_I, filter_II, prev = -999, crt = [1, 2, 3, 4, 5, 6]):
    X = prev
    if (h_pos - radius) <= x <= (h_pos + radius):
        C = (radius)**2 - (x - h_pos)**2  - (k_pos)**2
        i, j = Quadratic(1, -2*k_pos, -C)
        if i != 'Complex' and j != 'Complex':
            if i >= y >= j:
                if (filter_I >= crt[0] and filter_II >= crt[3]) or (filter_I >= crt[1] and filter_II >= crt[4]) or (filter_I <= crt[2] and filter_II >= crt[5]):
                    return int(1)
                else:
                    return int(0)             
    return X

df.drop(columns=['trajectory_n1', 'prev_distance', 'trajectory_n2', 'hash', 'prevexit_citystate', 'temp_citystate', 'city_yn', 'previous_x_exit',
                 'previous_y_exit', 'Delimiter', 'ey_1', 'ex_1', 'vmax', 'vmean', 'vmin', 'ez', 'ex', 'ey', 'time_entry',
                 'time_exit', 'time_', 'entry_minute'], axis = 1, inplace = True)

print(df.shape)
print(pd.isnull(df).sum() > 0)

##corrmat = df.corr()
##f, ax = plt.subplots(figsize=(11, 8))
##sns.heatmap(corrmat, vmax=1.0, fmt='.2f', square = True, annot = True)
##plt.show()

train = df.loc[~df.x_exit.isnull()].sort_index()
test = df.loc[df.x_exit.isnull()].sort_index()

##train['x'] = train['x_exit'].apply(lambda x: 1 if 3750901.5068 <= x <= 3770901.5068 else 0)
##train['y'] = train['y_exit'].apply(lambda x: 1 if -19268905.6133 <= x <= -19208905.6133 else 0)
##train['target'] = train['x'] + train['y']
##train['target'] = train['target'].apply(lambda x: 1 if x == 2 else 0)
##train = train.drop(columns=['x', 'y', 'x_exit', 'y_exit'])

corrmat = train.corr()
f, ax = plt.subplots(figsize=(11, 8))
sns.heatmap(corrmat, vmax=1.0, fmt='.2f', square = True, annot = True)
plt.show()

train.to_csv('train.csv', index = False)
test.to_csv('test.csv', index = False)
