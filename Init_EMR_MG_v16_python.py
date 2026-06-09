import numpy as np
import csv
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
csv_path    = os.path.abspath(os.path.join(current_dir, '..', 'Data', 'sidelec_roche_plate_csv.csv'))

# Constants
kB = 1.38065e-23        # (J/K) Boltzmann's constant
F = 96485               # (C/mol) Faraday's constant
R = 8.314               # (J/mol/K) Gas constant
q = 1.6e-19             # (C) Electronic charge

# Load data
sidelec_PV    = []
sidelec_conso = []

with open(csv_path, 'r') as csvfile:
# with open('./../sidelec_roche_plate_csv2.csv') as csvfile:
# with open('./../sidelec_roche_plate_csv_filter3.csv') as csvfile:  
    reader = csv.reader(csvfile, delimiter=';') # change contents to floats
    for row in reader: # each row is a list
        sidelec_PV.append(float(row[1]))
        sidelec_conso.append(float(row[2]))

def load_lut(file):
    # Charge le CSV, trie par puissance (colonne 0) et retourne l'array
    path = os.path.join(current_dir, file)
    data = np.loadtxt(path, delimiter=',')
    return data[data[:, 0].argsort()].T # .T pour séparer P et Eff direct

# Load configuration
LOAD = {
    'u_ref': 230,              # (V)
    'P_ref': sidelec_conso * 31,    # (W) Profile for 13 buildings / *20 pour avoir un vecteur assez long
    'Ts': 3600                 # (s) 1h
}

# Passive components
PSV = {
    'R_line': 30,              # (Ohm) from Tahim et al.
    'L_bat': 200e-3,           # (H) from Kong et al.
    'L_ely': 200e-3,           # (H)
    'L_FC': 200e-3             # (H)
}

# Simulation parameters
temps = np.linspace(0, 3600*24*365, 24*365+1)  # (s) Time vector for one year, sample time = 1h

SIM = {
    'Tstart': temps[0],
    'Tend': temps[-1],          # End at the end of 10 nights & 10 days
    'Ts': (temps[1]-temps[0])/60/4  # (s) 1s, it works with Tbus = 10s
}

# PV data
PV = {
    'P': sidelec_PV * 31   # (W) Sample time 1h
}

# Battery data
BAT = {
    'Q_bat': 72,                # Cell storage capacity [Ah]
    'Ccapacity': 72 * 3600,     # Cell equivalent capacity [F]
    'I_nom': 72,                # (A) Current 1C
    'v_cell_nom': 12,           # Cell nominal voltage
    'v_cell_min': 11.4,         # Cell minimal voltage at 80% DoD
    'r': 3.4e-3,                # Cell equivalent serial resistance [Ohm]
    'series_num': 45,           # Number of cells in series
    'parallel_num': 1,          # Number of branches in parallel
    'CAPEX':150,                # €/kWh
    'SoH_EoL' : 0.7,            # SoH quand on déclare la fin de vie (40% de perte de capacité)
    'eff' : 0.95
}

BAT['cost'] = 0.9 * BAT['CAPEX'] * BAT['series_num'] * BAT['parallel_num'] * BAT['Q_bat'] * BAT['v_cell_nom'] / 1000

# H2 tank data
H2_tank = {
    'P': 500e6,                 # (Pa) 500 bar in a H2 tank
    'P_ref': 1e6,               # (Pa) Reference pressure
    'V': 500e-3,                # (m3) Volume of a H2 tank
    'T': 298,                   # (K) Temperature
    'n': 500e6 * 500e-3 / (R * 298),  # (mol)
    'density': 30               # (kg/m3) Volumetric density
}

# FC & ELY data
FC = {
    'R': 0.001,                # (Ohm) Kong et al.
    'eff': 0.50,
    'n_series': 12,            # Number of cells in series
    'n_parallel': 1,           # Number of stacks in parallel
    'T': 273 + 60,             # (K) FC temperature
    'E_0': 1.23,               # (V) 57.5/53 from Bressel et al.
    'CAPEX':2500,              # €/kW
    'CAPEX_stack': 0.3*2500,   #€/kW on prend 30% du CAPEX total (qui inclut BoP etc.)
    'SoH_EoL' : 0.9,           # 20% de perte en tension
}

j_0  =  1e-3               # (A/cm2) Exchange current (Bressel thesis)
j_L  =  1                  # (A/cm2) Limiting current (Bressel thesis)
j_in = 0.0051              # (A/cm2) Internal current (Suyao model)
A    =  0.6e-4             # Tafel's constant
B    = -1.5e-4             # Concentration drop constant
S    = 220                 # (cm2) Electrode surface (Kong et al. + Bressel)

ELY = {
    'R': 0.001,              # (Ohm) Kong et al.
    'eff': 0.65,
    'n_series': 40,
    'n_parallel': 1,
    'T': 273 + 60,
    'E_0': 1.23,
    'CAPEX':2500,            #€ / kW
    'CAPEX_stack': 563,      #€/kW
    'SoH_EoL' : 0.9,         # 20% de perte de tension
}

FC['i_fc_max']   = 196.5598 * FC['n_parallel']
ELY['i_ely_max'] = 219.9 * ELY['n_parallel']

FC['P_fc_max']   = FC['i_fc_max'] * FC['n_series'] * (FC['E_0'] - FC['R'] * FC['i_fc_max'] / FC['n_parallel'] 
                                        - A * FC['T'] * np.log((FC['i_fc_max'] / S / FC['n_parallel'] + j_in) / j_0)
                                        - B * FC['T'] * np.log(1 - FC['i_fc_max'] / S / FC['n_parallel'] / j_L))
ELY['P_ely_max'] = ELY['i_ely_max'] * ELY['n_series'] * (ELY['E_0'] + ELY['R'] * ELY['i_ely_max'] / ELY['n_parallel'] 
                                           + A * ELY['T'] * np.log((ELY['i_ely_max'] / S / ELY['n_parallel'] + j_in) / j_0)
                                           + B * ELY['T'] * np.log(1 - ELY['i_ely_max'] / S / ELY['n_parallel'] / j_L))

FC['cost']  = FC['CAPEX_stack']  * FC['P_fc_max']   * 0.9 / 1000 #cout de remplacement avec 90% CAPEX stack
ELY['cost'] = ELY['CAPEX_stack'] * ELY['P_ely_max'] * 0.9 / 1000

# Bus DC data
BUS = {
    'u_ref_max': 410,          # (V) Arbitrary for now
    'u_ref_min': 390,          # (V) Arbitrary for now
    'u_init': 400,             # (V)
    'u_nom': 400,              # (V)
    'C': 1,                    # (F) Arbitrary, based on Suyao value
    'T_res': 60                # (s) Characteristic time
}

# DC/DC converters & Lines
CONV = {
    'eta': 0.9                # Efficiency
}

# EKF parameters
EKF = {
    'Ts': 10,
    'x0': np.array([0, 0]),  # State of health and degradation rate at t0
    'P00': np.zeros((2, 2)), # Error covariance at t0
    'A': np.array([[1, 10], [0, 1]]),
    'R': 1**2,                # (V^2) Measurement noise variance (Bressel thesis p95 pdf)
    'Q': np.diag([0, 1e-16])  # Process noise covariance
}
