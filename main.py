import math
import numpy as np
import matplotlib
import spiceypy
import pyoorb as oorb
import pandas as pd
import astropy
import astropy.units as u
import sbpy
from astropy.time import Time
from sbpy.data import Orbit, Ephem

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

# def turnDegToHrs(inp):
#     hrs = math.floor(inp / 15)
#     print("hours", hrs)
#     remainder = abs(15*hrs - inp)
#     mins = math.floor(remainder/15)
#     print("remainder", remainder)
#     remainder = abs(mins/60 - remainder)
#     secs = round(remainder)
#     return [hrs, mins, secs]

epoch_init = Time('2025-05-13T00:00:00', scale='utc')
epoch_calc = Time('2027-05-15T00:00:00', scale='utc')

ceres = orbitFromDict('Ceres', 'KEP', 2.77, 0.0786, 10.6, 73.6, 80.3, 188.7, epoch_init, 3.34, 0.15)
ceresPos = getCoordsAtEpoch(ceres, epoch_calc)

