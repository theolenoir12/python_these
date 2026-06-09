#on ré-écris le code de cost_fcn_total

import numpy as np
from scipy.signal import argrelextrema
from scipy.interpolate import interp1d
from .cost_fcn_total2 import *
from .Init_EMR_MG_v16_python import *

def get_soh(alpha_fc, P_fc, alpha_ely, P_ely, P_bat, SoC, SoH_bat):
    
    cost_bat = get_cost_bat(P_bat, SoC,SoH_bat)
    cost_fc  = get_cost_fc(alpha_fc, P_fc)[0]
    cost_ely = get_cost_ely(alpha_ely, P_ely)[0]
    
    SoH_bat_t = 1 - cost_bat / BAT['cost'] * (1 - BAT['SoH_EoL'])
    SoH_fc_t  = 1 - cost_fc / FC['cost'] * (1 - FC['SoH_EoL'])
    SoH_ely_t = 1 - cost_ely / ELY['cost'] * (1 - ELY['SoH_EoL'])
    
    # SoH_bat_t = SoH_bat_t - 8.8e-4/100 * len(P_bat) #Calendaire : 3% par an
    # SoH_fc_t  = SoH_fc_t - 1/(365*24) * len(P_fc) / 100    #Calendaire comme le idling
    # SoH_ely_t = SoH_ely_t - 1/100/(365*24) * len(P_ely)     #Calendaire : 2% pour 1000h
    
    return SoH_bat_t, SoH_fc_t, SoH_ely_t