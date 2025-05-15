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
import time

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
times = obs_df['observationStartMJD'].to_numpy()
obs_ra = obs_df['fieldRA'].to_numpy()
obs_dec = obs_df['fieldDec'].to_numpy()
mpc_dict = mpc_df.to_dict()
ssnamenrs = np.array(list(mpc_dict['Number'].values()))

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

start_time = time.time()

S = 0
N = 1

final = 202423

RA_temp = [0]*len(times[0:final])
DEC_temp = [0]*len(times[0:final])

for name in ssnamenrs[S:N]:
    elements = getOrbitalElements(mpc_df, name)
    epoch_init = Time(times[0:final], format='mjd')
    orbit_object = Orbit.from_dict(elements)
    coords = Ephem.from_oo(orbit_object, epoch_init)
    RA_temp.append(coords['RA'])
    DEC_temp.append(coords['DEC'])

end_time = time.time()

print(f"The function took {end_time-start_time} seconds to execute.")
