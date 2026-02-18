# IGNORE THIS FILE
import math

import matplotlib.pyplot as plt
import numpy as np

q = np.linspace(-3, 3, 40)
p = q
r = 40
np.random.seed(42)
y_train = q**4 + 2 * p**3 + np.random.normal(0, 16, size=q.shape)
y_test = p**4 + 2 * q**3 + np.random.normal(0, 16, size=p.shape)
gg = y_train


def b(com):
    ff = np.polyfit(q, gg, deg=com)
    pf = np.poly1d(ff)
    xf = np.linspace(-3, 3, r)
    yf = pf(xf)
    return yf


c = b
ff = c
best_fit = c
x = q
