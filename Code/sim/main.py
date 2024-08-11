from PIL import Image
import numpy as np
import pandas as pd
from scipy import integrate

# Load Data
LMS_sens = pd.read_csv('SPD2LMS.csv').to_numpy()
RGB_sens = pd.read_csv('SPD2RGB.csv').to_numpy()

# Cone fundamentals with or without shift
dL = 20
dM = 20
dS = 20
L = lambda wv: np.interp(wv+dL, LMS_sens[:,0], LMS_sens[:,1])
M = lambda wv: np.interp(wv+dM, LMS_sens[:,0], LMS_sens[:,2])
S = lambda wv: np.interp(wv+dS, LMS_sens[:,0], LMS_sens[:,3])
LMS = lambda wv: np.array([L(wv), M(wv), S(wv)])

# Converted to OPP fundamentals
# [L, M, S] -> [WS, RG, YB]
LMS2OPP = np.array([[0.600, 0.400, 0.000], 
                      [0.240, 0.105, -0.700], 
                      [1.200, -1.600, 0.400]])

WS = lambda wv: np.dot(LMS(wv), LMS2OPP[0])
RG = lambda wv: np.dot(LMS(wv), LMS2OPP[1])
YB = lambda wv: np.dot(LMS(wv), LMS2OPP[2])
OPP = lambda wv: np.array([WS(wv), RG(wv), YB(wv)])

# Create RGB -> OPP Mappings
R = lambda wv: np.interp(wv+dL, RGB_sens[:,0], RGB_sens[:,1])
G = lambda wv: np.interp(wv+dM, RGB_sens[:,0], RGB_sens[:,2])
B = lambda wv: np.interp(wv+dS, RGB_sens[:,0], RGB_sens[:,3])

WS_R, _ = integrate.quad(lambda wv: WS(wv)*R(wv), 390, 830)
WS_G, _ = integrate.quad(lambda wv: WS(wv)*G(wv), 390, 830)
WS_B, _ = integrate.quad(lambda wv: WS(wv)*B(wv), 390, 830)
RG_R, _ = integrate.quad(lambda wv: RG(wv)*R(wv), 390, 830)
RG_G, _ = integrate.quad(lambda wv: RG(wv)*G(wv), 390, 830)
RG_B, _ = integrate.quad(lambda wv: RG(wv)*B(wv), 390, 830)
YB_R, _ = integrate.quad(lambda wv: YB(wv)*R(wv), 390, 830)
YB_G, _ = integrate.quad(lambda wv: YB(wv)*G(wv), 390, 830)
YB_B, _ = integrate.quad(lambda wv: YB(wv)*B(wv), 390, 830)

WS_RGB = np.array([WS_R, WS_G, WS_B])
RG_RGB = np.array([RG_R, RG_G, RG_B])
YB_RGB = np.array([YB_R, YB_G, YB_B])

WS_RGB /= np.sum(WS_RGB)
RG_RGB /= np.sum(RG_RGB)
YB_RGB /= np.sum(YB_RGB)

print(WS_RGB)
print(RG_RGB)
print(YB_RGB)


"""
(141.7596268510127) (69.64227150458456) (4.137359407892389)
(55.315679360633176) (19.764701849776888) (-26.70093893254981)
(56.90194996115206) (-25.748482523415806) (10.888802405161481)

"""