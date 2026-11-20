import numpy as np;
import pandas as pd;
import matplotlib.pyplot as plt;
from scipy.stats import chi2, f;

RHO = 180 / np.pi * 3600;

# Task 1

initials = np.loadtxt("initials5.txt");
x0 = initials[:, 1:].reshape(1);
u = x0.size;


fixed = [0, 1, 8];
free = [i for i in range(u) if i not in fixed];

def read_epoch(dist_file, ang_file):
    dis = np.loadtxt(dist_file);
    ang = np.loadtxt(ang_file);


    dist_points = np.repeat
