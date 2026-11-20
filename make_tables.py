
import numpy as np;
import pandas as pd;
from lab9 import *;

# Task 1: Coordinate and residual tables

for epoch in (1, 2):

    # Select data for the current epoch
    if epoch == 1:
        xhat = xhat1;
        Cx = Cx1;
        r = r1;
        dist_pts = dist_pts1;
        ang_pts = ang_pts1;

    else:
        xhat = xhat2;
        Cx = Cx2;
        r = r2;
        dist_pts = dist_pts2;
        ang_pts = ang_pts2;

    # Coordinate table

    # Extract adjusted coordinates
    x_coords = xhat[0::2].round(4);
    y_coords = xhat[1::2].round(4);

    # Calculate coordinate standard deviations in mm
    std = np.sqrt(np.diag(Cx)) * 1000;
    sigma_x = std[0::2].round(2);
    sigma_y = std[1::2].round(2);

    # Create coordinate table
    coords = pd.DataFrame({
        "Point": range(1, 6),
        "X (m)": x_coords,
        "Y (m)": y_coords,
        "σX (mm)": sigma_x,
        "σY (mm)": sigma_y
    });

    coords.to_excel(f"epoch{epoch}_coordinates.xlsx", index=False);

    # Residual table

    # Separate distance and angle residuals
    n_dist = len(dist_pts);

    dist_res = r[:n_dist].reshape(-1, 3) * 1000;
    ang_res = r[n_dist:].reshape(-1, 3) * RHO;

    # Create observation labels
    labels = [];

    for i, j in dist_pts[::3]:
        labels.append(f"Distance {i}-{j}");

    for i, b, t in ang_pts[::3]:
        labels.append(f"Angle {b}-{i}-{t}");

    # Assign units to observations
    units = ["mm"] * len(dist_res);
    units += ["arcsec"] * len(ang_res);

    # Combine residuals into one matrix
    values = np.vstack([dist_res, ang_res]);
    values = values.round(2);

    # Create residual table
    residuals = pd.DataFrame({
        "Observation": labels,
        "Units": units,
        "Round 1": values[:, 0],
        "Round 2": values[:, 1],
        "Round 3": values[:, 2]
    });

    residuals.to_excel(f"epoch{epoch}_residuals.xlsx", index=False);


# Task 2: Global chi-square test

global_table = pd.DataFrame({
    "Epoch": [1, 2],
    "Chi-square": [round(T1, 4), round(T2, 4)],
    "Variance factor": [round(s2_1, 4), round(s2_2, 4)],
    "Lower bound": [round(low1, 4), round(low2, 4)],
    "Upper bound": [round(high1, 4), round(high2, 4)],
    "Result": [
        "PASS" if passed1 else "FAIL",
        "PASS" if passed2 else "FAIL"
    ]
});

global_table.to_excel("global_test.xlsx", index=False);


# Task 3: Bartlett's test

bartlett_passed = F_bartlett <= F_bartlett_critical;

if bartlett_passed:
    bartlett_result = "PASS";
else:
    bartlett_result = "FAIL";

bartlett_table = pd.DataFrame({
    "Test statistic": [round(F_bartlett, 4)],
    "Critical value": [round(F_bartlett_critical, 4)],
    "Result": [bartlett_result]
});

bartlett_table.to_excel("bartlett_test.xlsx", index=False);


# Task 4: Deformation analysis

# Separate X and Y displacement components
dx = d[0::2] * 1000;
dy = d[1::2] * 1000;

deformation_table = pd.DataFrame({
    "Point": range(1, 6),
    "dX (mm)": dx.round(2),
    "dY (mm)": dy.round(2)
});

deformation_table.to_excel("deformation.xlsx", index=False);


# Task 5: Global congruency test

gct_passed = y <= y_crit;

if gct_passed:
    gct_result = "PASS";
else:
    gct_result = "FAIL";

gct_table = pd.DataFrame({
    "Combined variance factor": [round(s2_c, 4)],
    "Rank of Qd": [r_d],
    "Test statistic": [round(y, 4)],
    "Critical value": [round(y_crit, 4)],
    "Result": [gct_result]
});

gct_table.to_excel("congruency_test.xlsx", index=False);
