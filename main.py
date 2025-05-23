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
# python main.py

# This is the epoch for ALL mpc asteroids. All of the MPC asteroids have this epoch, but it really shouldn't be hard-coded - need to fix
epoch_og = Time('2025-05-05T00:00:00', scale='utc')

# This function pulls the asteroid's RAs and DECs from the database, and then determines when it's in view of the LSST
# [astname] is the asteroid's ssnamenr
# [afterEpoch] is the starting time for the search (anything earlier than this time will be ignored)
#    - It must be an astropy Time object.
#    - Example: Time('2025-10-01T00:00:00', scale='utc')
# [d] is the maximum allowed difference (in days) between asteroid's observation time and LSST's snapshot time
#    - Decreasing [d] will shorten the calculation time, but may miss some LSST sightings.
#    - Because each asteroid's RA and DEC was generated 1 day apart, [d]<0.5 is recommended
def listWhenAstIsInLSST(astname, afterEpoch, d):
    start = time.time()
    afterEpoch = afterEpoch.mjd
    R = 1.75 # Degrees
    ast_df = pd.read_parquet('generated_coordinates.parquet',dtype_backend="pyarrow",columns=["ssnamenr","RA","DEC","observationMJD"],filters=[('ssnamenr', '=', astname)])
    end = time.time()
    print("Initialization took ",(end-start),"seconds")

    if(len(ast_df) == 0):
        print("ERROR LOADING ASTEROID'S DATA! SSNAMENR OF",astname,"NOT IN DATABASE")
    else:
        in_or_not = np.zeros(len(ast_df), dtype=bool)
        currentRA = 0
        currentDEC = 0
        currentRA_LSST = 0
        currentDEC_LSST = 0
        # Repeats over the length of the asteroid's RA/DEC list (800 right now)
        for i in range(len(ast_df)): # len(ast_df)
            currentEpoch = ast_df['observationMJD'][i]
            if(currentEpoch > afterEpoch): # Makes sure the observation MJD is not in the past
                currentRA = ast_df['RA'][i]
                currentDEC = ast_df['DEC'][i]

                upperBound = currentEpoch + d/2
                lowerBound = currentEpoch - d/2
                LSST_snapshots = pd.read_parquet('lsst_snapshots.parquet',dtype_backend="pyarrow",columns=["fieldRA","fieldDec","observationStartMJD"],filters=[('observationStartMJD','>=',lowerBound),('observationStartMJD','<=',upperBound)])
                for ii in range(len(LSST_snapshots)):
                    currentRA_LSST = LSST_snapshots['fieldRA'][ii]
                    currentDEC_LSST = LSST_snapshots['fieldDec'][ii]
                    P = math.sqrt(((currentRA - currentRA_LSST)* np.cos(currentDEC-currentDEC_LSST))**2 + (currentDEC-currentDEC_LSST)**2)
                    if P <= R:
                        in_or_not[i] = True
                        currentEpoch_LSST = LSST_snapshots['observationStartMJD'][ii]
                        print("Asteroid #",astname," will be in view of the LSST at MJD of",currentEpoch,"(asteroid time)")
                    else:
                        in_or_not[i] = False
            else:
                print(currentEpoch,"is in the past: skipping")
    end = time.time()
    print("DONE! Time:",(end-start),"seconds")

epoch_current = Time('2025-10-01T00:00:00', scale='utc') # The function will ignore any observations before this date
listWhenAstIsInLSST(35, epoch_current, 0.4)

# def convertToOrbitType(targetname, a, e, i, Node, Peri, M, G, H, epoch_og):
#     return Orbit.from_dict({''
#     'targetname': targetname,
#     'orbtype': 'KEP',
#     'a': a * u.au,
#     'e': e,
#     'i': i * u.deg,
#     'w': Peri * u.deg,
#     'Omega': Node * u.deg,
#     'M': M * u.deg,
#     'epoch': epoch_og,
#     'H': H * u.mag,
#     'G': G});

# def getCoordsAtEpoch(orbit, epoch):
#     epoch = Time(epoch,format='mjd')
#     epoch = epoch + np.arange(1) * u.day
#     coords = Ephem.from_oo(orbit, epoch)
#     o_RA = coords['RA'].value
#     o_DEC = coords['DEC'].value
#     return {"RA":o_RA, "DEC":o_DEC};

# Returns orbital elements of a specific asteroid given an asteroid ID
# Runs under the assumption that the dataset named 'df' exists
# def getOrbitalElements(asteroidNameNumber):
#     # df is the pandas dataframe already preprocessed (in this case)
#     # d is the dictionary of said dataframe
#     d = df.to_dict()
#     nr = asteroidNameNumber

#     number = d['Number'][list(d['Number'].values()).index(nr)]
#     a = d['a'][list(d['Number'].values()).index(nr)]
#     e = d['e'][list(d['Number'].values()).index(nr)]
#     i = d['i'][list(d['Number'].values()).index(nr)]
#     Node = d['Node'][list(d['Number'].values()).index(nr)]
#     Peri = d['Peri'][list(d['Number'].values()).index(nr)]
#     M = d['M'][list(d['Number'].values()).index(nr)]
#     G = d['G'][list(d['Number'].values()).index(nr)]
#     H = d['H'][list(d['Number'].values()).index(nr)]
#     epoch = Time('2025-10-05T00:00:00', scale='utc') # TODO: ACTUALLY PULL AND CONVERT THE EPOCH TIMES
#     return convertToOrbitType(number, a, e, i, Node, Peri, M, G, H, epoch);

# ---------- THIS FUNCTION IS MOSTLY JUST FOR FUN ----------
# ---------- IT'S NOT NEEDED FOR THE MONSOON CALCULATIONS ----------
# startDate, endDate must be in MJD
# def generatePrettyPlotOfAsteroid(name, startDate, endDate, stepCount, plotName):
#     plt.figure(figsize=(10, 6))
#     plt.subplot(projection="aitoff")

#     RA_temp_plot = []
#     DEC_temp_plot = []
#     generatedJDs = []
#     RA_temp = []
#     DEC_temp = []
#     diff = endDate - startDate

#     i_temp = 0
#     for i in range(startDate, endDate, stepCount):
#         generatedJDs.append(Time(i,format='mjd'))
#         i_temp = i_temp+1

#     orbit_ast = getOrbitalElements(name) # Gets the orbital data of the asteroid

#     for i in range(0, len(generatedJDs)):
#         coords_temp = getCoordsAtEpoch(orbit_ast, generatedJDs[i])
#         RA_temp.append(coords_temp['RA'][0])
#         DEC_temp.append(coords_temp['DEC'][0])

#     RA_temp_plot = np.zeros(len(RA_temp))
#     DEC_temp_plot = np.zeros(len(DEC_temp))

#     for i in range(len(RA_temp)):
#         DEC_temp_plot[i] = DEC_temp[i] * (np.pi / 180.0)
#         if RA_temp[i] > 180.0:
#             RA_temp_plot[i] = RA_temp[i]*np.pi/180.0 - 2*np.pi
#         else:
#             RA_temp_plot[i] = RA_temp[i]*np.pi / 180.0

#     plt.plot(RA_temp_plot, DEC_temp_plot, 'o', color='black', markersize=3, alpha=0.2)
#     plt.xlabel('RA')
#     plt.ylabel('DEC')
#     plt.title(("Asteroid #",name))
#     plt.grid(axis='x')
#     plt.grid(axis='y')
#     plt.savefig(plotName)
#     plt.grid(True)
#     plt.show()
