import numpy as np

def Polar2Complex(abs_val, angle):
    return abs_val * np.exp(1j * angle)