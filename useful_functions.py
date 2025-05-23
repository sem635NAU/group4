import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import spiceypy
import pyoorb as oorb
import pyarrow
import pandas as pd
import astropy
import astropy.units as u
import sbpy
import pymongo
import time
from pymongo import MongoClient
from astropy.time import Time
from astropy.coordinates import SkyCoord
from sbpy.data import Orbit, Ephem

import warnings
warnings.filterwarnings('ignore')

# TO RUN ME:
# cd .conda/envs/find_orbits
# conda activate find_orbits
# python useful_functions.py

df = pd.read_csv('filtered_mpc_df.csv')

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
    epoch = Time(epoch,format='mjd')
    epoch = epoch + np.arange(1) * u.day
    coords = Ephem.from_oo(orbit, epoch)
    o_RA = coords['RA'].value
    o_DEC = coords['DEC'].value
    return {"RA":o_RA, "DEC":o_DEC};

# Returns orbital elements of a specific asteroid given an asteroid ID
# Runs under the assumption that the dataset named 'df' exists
def getOrbitalElements(df1, asteroidNameNumber):
    # df is the pandas dataframe already preprocessed (in this case)
    # d is the dictionary of said dataframe
    d = df1.to_dict()
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

# ---------- THIS FUNCTION IS MOSTLY JUST FOR FUN ----------
# ---------- IT'S NOT NEEDED FOR THE MONSOON CALCULATIONS ----------
# startDate, endDate must be in MJD
def generatePrettyPlotOfAsteroid(name, startDate, endDate, stepCount, plotName):
    plt.figure(figsize=(10, 6))
    plt.subplot(projection="aitoff")

    RA_temp_plot = []
    DEC_temp_plot = []
    generatedJDs = []
    RA_temp = []
    DEC_temp = []
    diff = endDate - startDate

    i_temp = 0
    for i in range(startDate, endDate, stepCount):
        generatedJDs.append(Time(i,format='mjd'))
        i_temp = i_temp+1

    orbit_ast = getOrbitalElements(df, name) # Gets the orbital data of the asteroid

    for i in range(0, len(generatedJDs)):
        coords_temp = getCoordsAtEpoch(orbit_ast, generatedJDs[i])
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
    plt.grid(axis='y')
    plt.savefig(plotName)
    plt.grid(True)
    plt.show()