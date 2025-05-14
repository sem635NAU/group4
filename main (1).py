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

# TO RUN ME:
# cd .conda/envs/find_orbits
# conda activate find_orbits
# python main.py

pmcli_lsst = pymongo.MongoClient('mongodb://group4:password@cmp4818.computers.nau.edu:27018/')
LSST_db = pmcli_lsst['group4']['observations']

def orbitFromDict(targetname, orbtype, a, e, i, w, omega, M, epoch_og, H, G):
    return Orbit.from_dict({''
    'targetname': targetname,
    'orbtype': orbtype,
    'a': a * u.au,
    'e': e,
    'i': i * u.deg,
    'w': w * u.deg,
    'Omega': omega * u.deg,
    'M': M * u.deg,
    'epoch': epoch_og,
    'H': H * u.mag,
    'G': G});

def getCoordsAtEpoch(orbit, epoch):
    epoch = epoch + np.arange(1) * u.day
    coords = Ephem.from_oo(orbit, epoch)
    print(coords)
    o_RA = coords['RA'].value
    o_DEC = coords['DEC'].value
    return {"RA":o_RA, "DEC":o_DEC};


# TEMPORARY SHIT
# This is a temporary script that generates a list of JD values for Ceres

epoch_init = Time('2025-05-13T00:00:00', scale='utc')
# epoch_calc = Time('2027-05-15T00:00:00', scale='utc')

ceres = orbitFromDict('Ceres', 'KEP', 2.77, 0.0786, 10.6, 73.6, 80.3, 188.7, epoch_init, 3.34, 0.15)
# ceresPos = getCoordsAtEpoch(ceres, epoch_calc)

generatedJDs = []
for i in range(0, 100, 1):
    generatedJDs.append(Time(60810 + (5*i),format='mjd'))
# print(generatedJDs)

plt.figure(figsize=(10, 6))
ceresGeneratedJDs = []
RA_temp = []
DEC_temp = []
for i in range(0, 100, 5):
    epoch_calc_temp = generatedJDs[i]
    coords_temp = getCoordsAtEpoch(ceres, epoch_calc_temp)
    RA_temp.append(coords_temp['RA'][0])
    DEC_temp.append(coords_temp['DEC'][0])

plt.scatter(RA_temp, DEC_temp)
plt.xlabel('Data Values')
plt.ylabel('Frequency')
plt.title('Dot Plot')
plt.grid(axis='x')
plt.show()