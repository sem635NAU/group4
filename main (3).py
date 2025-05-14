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

# pmcli = pymongo.MongoClient('mongodb://group4:password@cmp4818.computers.nau.edu:27018/')
# db = pmcli['ztf']['snapshot 1']
# ssnamenr = db.find({},{'ssnamenr':1, '_id':0})
# ssnamenr = list(ssnamenr)
# unique_ssnamenr = set()
# for s in ssnamenr:
#     unique_ssnamenr.add(s['ssnamenr'])
# asteroids = pd.DataFrame(unique_ssnamenr)

# ztf_mpc_db = pmcli['ztf']['ztf_mpc_temp']
# mpc_data = ztf_mpc_db.find({},{'_id':0, 'a':1, 'e':1, 'i':1, 'Peri':1, 'Node':1, 'M':1, 'H':1, 'G':1, 'Number':1})
# mpc_df = pd.DataFrame(mpc_data)

# w = argument of periapsis (I think)
# Omega = node
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
    # print(coords)
    o_RA = coords['RA'].value
    o_DEC = coords['DEC'].value
    return {"RA":o_RA, "DEC":o_DEC};

epoch_init = Time(60808, format='mjd')
epoch_calc = Time('2025-05-05T00:00:00', scale='utc')
epoch_TEST = Time('2026-10-15T00:00:00', scale='utc')

# POTENTIAL PROBLEM: THE ORBITAL TYPE IS 'MB II' FOR PALLAS
# ceres = orbitFromDict('Ceres', 'KEP', 2.77, 0.0786, 10.6, 73.6, 80.3, 188.7, epoch_calc, 3.34, 0.15) REAL CERES
ceres = orbitFromDict('Pallas', 'KEP', 2.7701937, 0.2305404, 34.92402, 310.91037, 172.8953, 168.79869, epoch_init, 4.11, 0.15) # THIS IS PALLAS
ceresPos = getCoordsAtEpoch(ceres, epoch_TEST)
print ("CERES POS FOR", epoch_TEST)
print(ceresPos["RA"])
print(ceresPos["DEC"])

day_span = 300
jump_by = 10

generatedJDs = []
for i in range(0, day_span, 1):
    generatedJDs.append(Time(60800 + (jump_by*i),format='mjd'))
# print(generatedJDs)

plt.figure(figsize=(10, 6))
ceresGeneratedJDs = []
RA_temp = []
DEC_temp = []

for i in range(0, day_span, 1):
    epoch_calc_temp = generatedJDs[i]
    coords_temp = getCoordsAtEpoch(ceres, epoch_calc_temp)
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

plt.subplot(projection="aitoff")
plt.plot(RA_temp_plot, DEC_temp_plot, 'o', color='black', markersize=3, alpha=0.2)
plt.xlabel('RA')
plt.ylabel('DEC')
# plt.title('CERES')
plt.title('PALLAS')
plt.grid(axis='x')
plt.savefig('test1.png')
plt.show()