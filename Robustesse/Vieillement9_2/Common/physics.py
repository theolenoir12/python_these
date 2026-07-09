import numpy as np

# Common/physics.py
def get_p_max_fc(alpha_fc_t, FC, constants):
    
    i_fc_max_t = (-194.3950 * alpha_fc_t + 196.5598)
    A, B, S, j_in, j_0, j_L = constants
    P_fc_max_t = i_fc_max_t * FC['n_parallel'] * FC['n_series'] * (FC['E_0'] - FC['R'] * (1 + alpha_fc_t) * i_fc_max_t / FC['n_parallel'] 
                                            - A * FC['T'] * np.log((i_fc_max_t / S / FC['n_parallel'] + j_in) / j_0)
                                            - B * FC['T'] * np.log(1 - i_fc_max_t / S / FC['n_parallel'] / j_L / (1 - alpha_fc_t)))
    return P_fc_max_t

def get_p_max_ely(alpha_ely_t, ELY, constants):
    i_ely_max_t = (-219.9 * alpha_ely_t + 219.9)    
    A, B, S, j_in, j_0, j_L = constants
    P_ely_max_t = i_ely_max_t * ELY['n_parallel'] * ELY['n_series'] * (ELY['E_0'] + ELY['R'] * (1 + alpha_ely_t) * i_ely_max_t / ELY['n_parallel'] 
                                               + A * ELY['T'] * np.log((i_ely_max_t / S / ELY['n_parallel'] + j_in) / j_0)
                                               + B * ELY['T'] * np.log(1 - i_ely_max_t / S / ELY['n_parallel'] / j_L / (1 - alpha_ely_t)))
    return P_ely_max_t