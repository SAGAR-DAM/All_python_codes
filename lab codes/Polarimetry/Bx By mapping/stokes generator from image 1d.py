# -*- coding: utf-8 -*-
"""
Created on Sun Dec 31 12:07:14 2023

@author: sagar
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from skimage import io, draw

import magfield.magmodule_sagar as mms

#################################################
"""     Give the pump-probe image  """
#################################################
Pu_Pr_DP = io.imread("D:\\data Lab\\anandam data\\dec 2023\\27th dec 2023\\good data\\Aluminium\\P D\\pump probe\\P_015.tif")
Pu_Pr_QT = io.imread("D:\\data Lab\\anandam data\\dec 2023\\27th dec 2023\\good data\\Aluminium\\T Q\\pump probe\\C_015.tif")


#################################################
"""  BS + Filter factors at experiment time  """
#################################################
P_filter = 792.82
D_filter = 415.2
T_filter = 1543.17
Q_filter = 666.67



#################################################
"""      Give the Pump only image """
#################################################
pump_only_DP = io.imread("D:\\data Lab\\anandam data\\dec 2023\\27th dec 2023\\good data\\Aluminium\\P D\\pump only\\P_002.tif")
pump_only_QT = io.imread("D:\\data Lab\\anandam data\\dec 2023\\27th dec 2023\\good data\\Aluminium\\T Q\\pump only\\C_002.tif")

bg_fac_pump_only_DP = 1.2
bg_fac_pump_only_QT = 1.05

#################################################

tol = 0.01


#################################################
#################################################
#################################################
#################################################



pump_only_D , pump_only_P = mms.get_image_tl_and_br(image = pump_only_DP, background_factor = bg_fac_pump_only_DP)
pump_only_Q , pump_only_T = mms.get_image_tl_and_br(image = pump_only_QT, background_factor = bg_fac_pump_only_QT)

D_noise = mms.get_nonzero_value_average(pump_only_D)
P_noise = mms.get_nonzero_value_average(pump_only_P)
Q_noise = mms.get_nonzero_value_average(pump_only_Q)
T_noise = mms.get_nonzero_value_average(pump_only_T)


P_int, D_int, Q_int, T_int, s, mod_s, background_factor = mms.stokes_generator( image_P = Pu_Pr_DP,
                                                                                image_C = Pu_Pr_QT,
                                                                                p_filter = P_filter,
                                                                                d_filter = D_filter,
                                                                                t_filter = T_filter,
                                                                                q_filter = Q_filter,
                                                                                d_image_pump_noise = D_noise,
                                                                                p_image_pump_noise = P_noise,
                                                                                t_image_pump_noise = T_noise,
                                                                                q_image_pump_noise = Q_noise,
                                                                                tol = tol)
print(f"s:  {s}")
print(f"mods: {mod_s}")
print(f"P signal integrated: {P_int}")
print(f"D signal integrated: {D_int}")
print(f"T signal integrated: {T_int}")
print(f"Q signal integrated: {Q_int}")
print(f"background factor: {background_factor}")
