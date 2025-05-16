import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import spiceypy
import pyoorb as oorb
import pandas as pd
#import astropy
import astropy.units as u
from pymongo import MongoClient
from astropy.time import Time
from sbpy.data import Orbit, Ephem
# import time

import warnings
warnings.filterwarnings('ignore')

# TO RUN ME:
# cd .conda/envs/find_orbits
# module load anaconda3
# conda activate find_orbits
# python main.py

# Bring in all the data
mpc_df = pd.read_csv('mpc_df.csv')
obs_df = pd.read_csv('midnight_in_Chile_df.csv')
interesting_asteroids = pd.read_csv('interesting_asteroids_df.csv')

# Make into useable lists
times = obs_df['Midnight in Chile'].to_numpy()
#obs_ra = obs_df['fieldRA'].to_numpy()
#obs_dec = obs_df['fieldDec'].to_numpy()
mpc_dict = mpc_df.to_dict()
ssnamenrs = interesting_asteroids["Interesting Asteroids"].to_numpy()

uri = 'mongodb://group4:password@cmp4818.computers.nau.edu:27018/'
DB_NAME = "group4"
COL_NAME = 'asteroid_positions'
src_client = MongoClient(uri)
src_db = src_client[DB_NAME]
col_connection = src_db[COL_NAME]

def pushToMongo(name, ra, dec, jd):
    try:
        col_connection.insert_one({'ssnamenr':name, 'RA':ra, 'DEC':dec, 'observationMJD':jd})
    except Exception as e:
        print(e)

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

# start_time = time.time()

S = 0 # Exclude asteroids '1', '2', '3', '4', and '5' since they are already done !
N = -1

final = -1 # for first week

RA_temp = [0]*len(times[0:final])
DEC_temp = [0]*len(times[0:final])

for name in ssnamenrs[S:N]:
    elements = getOrbitalElements(mpc_df, name)
    epoch_init = Time(times[0:final], format='mjd')
    orbit_object = Orbit.from_dict(elements)
    coords = Ephem.from_oo(orbit_object, epoch_init)
    RA_temp.append(coords['RA'])
    DEC_temp.append(coords['DEC'])
#    for RA, DEC, cur_jd in zip(coords['RA'], coords['DEC'],times[0:final]):
#        pushToMongo(str(name), float(RA.to_value(u.deg)), float(DEC.to_value(u.deg)), float(cur_jd))

# end_time = time.time()

print(f"The function took {end_time-start_time} seconds to execute.")
