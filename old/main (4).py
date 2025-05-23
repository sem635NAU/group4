import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import spiceypy
import pyoorb as oorb
import pandas as pd
import astropy
import astropy.units as u
import sbpy
import pymongo
from astropy.time import Time
from sbpy.data import Orbit, Ephem

import warnings
warnings.filterwarnings('ignore')

epoch_og = Time('2025-05-05T00:00:00', scale='utc') # This is the epoch for ALL mpc asteroids

# TO RUN ME:
# cd .conda/envs/find_orbits
# conda activate find_orbits
# python main.py

# This is the (pre-processed) data file (name is mpc_df.csv)
# ORDER:
# H,G,M,Peri,Node,i,e,a,Number
global df
df = pd.read_csv('mpc_df.csv')

def convertToOrbitType(targetname, a, e, i, Node, Peri, M, G, H, epoch_og):
    return Orbit.from_dict({''
    'targetname': targetname,
    'orbtype': 'KEP',
    'a': a * u.au,
    'e': e,
    'i': i * u.deg,
    'w': Peri * u.deg,
    'Omega': Node * u.deg,
    'M': M * u.deg,
    'epoch': epoch_og,
    'H': H * u.mag,
    'G': G});

def getCoordsAtEpoch(orbit, epoch):
    epoch = epoch + np.arange(1) * u.day
    coords = Ephem.from_oo(orbit, epoch)
    # print(coords)
    o_RA = coords['RA'].value
    o_DEC = coords['DEC'].value
    return {"RA":o_RA, "DEC":o_DEC};

# Returns orbital elements of a specific asteroid given an asteroid ID
# Runs under the assumption that the dataset named 'df' exists
def getOrbitalElements(asteroidNameNumber):
    # df is the pandas dataframe already preprocessed (in this case)
    # d is the dictionary of said dataframe
    d = df.to_dict()
    nr = asteroidNameNumber

    number = d['Number'][list(d['Number'].values()).index(nr)]
    a = d['a'][list(d['Number'].values()).index(nr)]
    e = d['e'][list(d['Number'].values()).index(nr)]
    i = d['i'][list(d['Number'].values()).index(nr)]
    Node = d['Node'][list(d['Number'].values()).index(nr)]
    Peri = d['Peri'][list(d['Number'].values()).index(nr)]
    M = d['M'][list(d['Number'].values()).index(nr)]
    G = d['G'][list(d['Number'].values()).index(nr)]
    H = d['H'][list(d['Number'].values()).index(nr)]
    epoch = Time('2025-10-05T00:00:00', scale='utc') # TODO: ACTUALLY PULL AND CONVERT THE EPOCH TIMES
    return convertToOrbitType(number, a, e, i, Node, Peri, M, G, H, epoch);

# Mostly just for fun
# startDate, endDate must be in MJD
def generatePrettyPlotOfAsteroid(name, startDate, endDate, stepCount, plotName):
    plt.figure(figsize=(10, 6))
    plt.subplot(projection="aitoff")

    RA_temp_plot = []
    DEC_temp_plot = []
    generatedJDs = []
    diff = endDate - startDate

    for i in range(0, diff):
        generatedJDs.append(Time(startDate + (stepCount*i),format='mjd'))

    for i in range(0, len(generatedJDs)):
        coords_temp = getCoordsAtEpoch(ceres, generatedJDs[i])
        RA_temp.append(coords_temp['RA'][0])
        DEC_temp.append(coords_temp['DEC'][0])

    RA_temp_plot = np.zeros(len(RA_temp))
    DEC_temp_plot = np.zeros(len(DEC_temp))

    for i in range(len(RA_temp)):
        DEC_temp_plot[i] = DEC_temp[i] * (np.pi / 180.0)
        if RA_temp[i] > 180.0:
            RA_temp_plot[i] = RA_temp[i]*np.pi/180.0 - 2*np.pi
        else:
            RA_temp_plot[i] = RA_temp[i]*np.pi / 180.0

    plt.plot(RA_temp_plot, DEC_temp_plot, 'o', color='black', markersize=3, alpha=0.2)
    plt.xlabel('RA')
    plt.ylabel('DEC')
    plt.title(("Asteroid #",name))
    plt.grid(axis='x')
    plt.savefig(plotName)
    plt.show()
generatePrettyPlotOfAsteroid(1, 60800, 61800, 5, "test2.png")

epoch_calc = Time('2025-10-05T00:00:00', scale='utc')
for name in range(1,3):
    orbit_temp = getOrbitalElements(name)
    print("orbit_temp:",orbit_temp)
    predict_tempt = getCoordsAtEpoch(orbit_temp,epoch_calc)
    print("PREDICTED RA DEC FOR ASTEROID #",name)
    print(predict_tempt)

# for name in mpc_df(['Number']).to_numpy():
    # dict_temp = 
