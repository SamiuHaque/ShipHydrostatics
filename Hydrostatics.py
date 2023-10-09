import numpy as np
import pandas as pd
import scipy as sp


# Reading the offset Table from text file
offsetTable = np.array(pd.read_csv("OffsetTable.txt", sep="\t", index_col=0, header=0))

# Ship Principle Particulars
Length = 146
Breadth = 22.5
Depth = 13.5
Draft = 9
Density = 1.025

# No. of Stations and Waterlines
station = len(offsetTable)
waterline = len(offsetTable[0])

# Determine Station and Waterline Spacing
stationSpacing = Length/(station-1)
waterlineSpacing = Draft/(waterline-1)

# Set two forms of offset table
stCol = offsetTable.copy()          # Waterline as Column
wlCol = stCol.transpose()           # Station as Column

# Array for taking moments
momentWl = np.arange(0, waterline)*waterlineSpacing
momentSt = np.arange(0, station)*stationSpacing

# Water-plane Area Calculation
wpArea = 2*sp.integrate.simps(wlCol, dx=stationSpacing)

# Integration in Vertical Direction
csArea = []
Volume = []
KB = []
for i in range(2, waterline+1):
    area = 2 * sp.integrate.simps(stCol[:, :i], dx=waterlineSpacing)
    csArea = np.hstack((csArea, area))      # Cross-Sectional Area for all waterlines(0 to end)
    vol = sp.integrate.simps(wpArea[:i], dx=waterlineSpacing)
    Volume = np.hstack((Volume, vol))       # Volume for all waterlines(0 to end)
    kb = (sp.integrate.simps(wpArea[:i]*momentWl[0:i], dx=waterlineSpacing))/vol
    KB = np.hstack((KB, kb))                # Vertical Center of Buoyancy for all waterlines(0 to end)
                                            # Ref. Point for VCB/KB is Keel
csArea = csArea.reshape(waterline-1, -1)

# Center of Floatation Calculation (Ref. Point Aft Perpendicular)
CF = (2*sp.integrate.simps(wlCol*momentSt, dx=stationSpacing))/wpArea
# TPC Calculation
TPC = wpArea*Density/100
# 2nd moment of Water-plan area about Centerline
I_cl = (2/3)*sp.integrate.simps(np.power(wlCol, 3), dx=stationSpacing)
I_cl = np.delete(I_cl, 0)
# BM_t calculation, BM_t = I_cl/Volume
BM_t = I_cl/Volume
# 2nd moment of Water-plan area about CF
I_AP = 2 * sp.integrate.simps(wlCol * np.power(momentSt, 2), dx=stationSpacing)
I_AP = np.delete(I_AP, 0)   # About Aft Perpendicular
I_CF = I_AP - wpArea[1:]*np.power(CF[1:], 2)
# BM_l calculation, BM_l = I_CF/Volume
BM_l = I_CF/Volume
# LCB Calculation (Ref. Point AP)
LCB = (sp.integrate.simps(csArea*momentSt, dx=stationSpacing))/Volume
