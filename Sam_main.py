import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import spiceypy
import pyoorb as oorb
import pandas as pd
#import astropy
import astropy.units as u
import pymongo
from astropy.time import Time
from sbpy.data import Orbit, Ephem

import warnings
warnings.filterwarnings('ignore')

# TO RUN ME:
# cd .conda/envs/find_orbits
# module load anaconda3
# conda activate find_orbits
# python main.py

# Bring in all the data
mpc_df = pd.read_csv('mpc_df.csv')
obs_df = pd.read_csv('obs_df.csv')

# Make into useable lists
time = obs_df['observationStartMJD'].to_numpy()
obs_ra = obs_df['fieldRA'].to_numpy()
obs_dec = obs_df['fieldDec'].to_numpy()
mpc_dict = mpc_df.to_dict()
ssnamenrs = np.array(list(mpc_dict['Number'].values()))

# def orbitFromDict(args):
# Orbit.from_dict({''
#                         # 'targetname': 'Ceres',
#                         'orbtype': 'KEP',
#                         'a': 2.77 * u.au,
#                         'e': 0.0786,
#                         'i': 10.6 * u.deg,
#                         'w': 73.6 * u.deg,
#                         'Omega': 80.3 * u.deg,
#                         'M': 188.7 * u.deg,
#                         'epoch': epoch_init,
#                         'H': 3.34 * u.mag,
#                         'G': 0.15})
#     [nr, a, e, i, Peri, Node, M, Epoch, H, G]
#     return Orbit.from_dict(args);

def getOrbitalElements(df1, asteroidNameNumber):
    # df1 is the pandas dataframe already preprocessed
    # d is the dictionary of said dataframe
    d = df1.to_dict()
    nr = asteroidNameNumber
    
    a = d['a'][list(d['Number'].values()).index(nr)]
    e = d['e'][list(d['Number'].values()).index(nr)]
    i = d['i'][list(d['Number'].values()).index(nr)]
    Node = d['Node'][list(d['Number'].values()).index(nr)]
    Peri = d['Peri'][list(d['Number'].values()).index(nr)]
    M = d['M'][list(d['Number'].values()).index(nr)]
    G = d['G'][list(d['Number'].values()).index(nr)]
    H = d['H'][list(d['Number'].values()).index(nr)]
    Epoch = d['Epoch'][list(d['Number'].values()).index(nr)]

    return {'targetname':nr,
            'orbtype':'KEP',
            'a':a * u.au, 
            'e':e, 
            'i':i * u.deg, 
            'w':Peri * u.deg, 
            'Omega':Node * u.deg, 
            'M':M * u.deg, 
            'epoch':Time(Epoch, format='jd'), 
            'H':H * u.mag, 
            'G':G}

def getCoordsAtEpoch(orbit, epoch):
    epoch = epoch + np.arange(1) * u.day
    coords = Ephem.from_oo(orbit, epoch)
    # print(coords)
    o_RA = coords['RA'].value
    o_DEC = coords['DEC'].value
    return epoch, {"RA":o_RA, "DEC":o_DEC}


S = 0
N = 1

for name in ssnamenrs[S:N]:
    X = getOrbitalElements(mpc_df, name)
    epoch_init = Time(time[0:100], format='mjd')
    print(time[0])
    orbit_object = Orbit.from_dict(X)
    coords = Ephem.from_oo(orbit_object, epoch_init)
    print(coords)

# POTENTIAL PROBLEM: THE ORBITAL TYPE IS 'MB II' FOR PALLAS
# ceres = orbitFromDict('Ceres', 'KEP', 2.77, 0.0786, 10.6, 73.6, 80.3, 188.7, epoch_calc, 3.34, 0.15) REAL CERES
#ceres = orbitFromDict('Pallas', 'KEP', 2.7701937, 0.2305404, 34.92402, 310.91037, 172.8953, 168.79869, epoch_init, 4.11, 0.15) # THIS IS PALLAS
#ceresPos = getCoordsAtEpoch(ceres, epoch_TEST)
#print ("CERES POS FOR", epoch_TEST)
#print(ceresPos["RA"])
#print(ceresPos["DEC"])

day_span = 300
jump_by = 10

generatedJDs = []
for i in range(0, day_span, 1):
    generatedJDs.append(Time(60800 + (jump_by*i),format='mjd'))
# print(generatedJDs)

#for i in range(0, day_span, 1):
#    epoch_calc_temp = generatedJDs[i]
#    coords_temp = getCoordsAtEpoch(ceres, epoch_calc_temp)
#    RA_temp.append(coords_temp['RA'][0])
#    DEC_temp.append(coords_temp['DEC'][0])

