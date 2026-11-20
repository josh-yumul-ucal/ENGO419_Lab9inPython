import numpy as np;
import pandas as pd;
from scipy.stats import chi2, f;

RHO = 180 / np.pi * 3600;

# Task 1

initials = np.loadtxt("initials5.txt");
x0 = initials[:, 1:].reshape(-1);
u = x0.size;


fixed = [0, 1, 8];
free = [i for i in range(u) if i not in fixed];

def read_epoch(dist_file, ang_file):
    dist = np.loadtxt(dist_file);
    ang = np.loadtxt(ang_file);

    # repeats the point_ids of each point three times to make room for three rounds of obs
    dist_pts = np.repeat(dist[:, :2].astype(int), 3, axis=0);
    dist_obs = dist[:, [2, 4, 5]].reshape(-1);
    dist_sigma = np.repeat(dist[:, 3], 3);

    # same for angle obs
    ang_pts = np.repeat(ang[:, :3].astype(int), 3, axis=0);
    angles = [];

    for c in (3, 7, 10):
        degrees = ang[:, c];
        minutes = ang[:, c + 1];
        seconds = ang[:, c + 2];

        decimal_degrees = degrees + minutes / 60 + seconds / 3600;

        angles.append(decimal_degrees);

    ang_deg = np.column_stack(angles);
    ang_obs = np.deg2rad(ang_deg.reshape(-1));

    ang_sigma = np.full(len(ang_obs), 2 / RHO);

    # Ensure that precisions match lab document
    assert np.allclose(dist[:, 3], 0.004) and np.allclose(ang[:, 6], 2);

    l = np.concatenate([dist_obs, ang_obs]);
    Cl = np.diag(np.concatenate([dist_sigma, ang_sigma]) ** 2);

    return l, Cl, dist_pts, ang_pts;

def design(x, dist_pts, ang_pts):
    n = len(dist_pts) + len(ang_pts);
    fx = np.zeros(n);
    A = np.zeros((n, u));

    for k, (i, j) in enumerate(dist_pts):
        dx = x[2 * j - 2] - x[2 * i - 2];
        dy = x[2 * j - 1] - x[2 * i -1];
        s = np.hypot(dx, dy);

        fx[k] = s;
        A[k, 2 * i - 2:2 * i] = [-dx / s, -dy / s];
        A[k, 2 * j - 2:2 * j] = [dx / s, dy / s];
    
    for k, (i, b, t) in enumerate(ang_pts, start=len(dist_pts)):
        for p, sign in ((t, 1), (b, -1)):
            dx = x[2 * p - 2] - x[2 * i - 2];
            dy = x[2 * p - 1] - x[2 * i - 1];
            s2 = dx**2 + dy**2;

            fx[k] += sign * np.arctan2(dx, dy);
            A[k, 2 * i - 2:2 * i] += sign * np.array([-dy, dx]) / s2;
            A[k, 2 * p - 2:2 * p] += sign * np.array([dy, -dx]) / s2;

    return fx, A;

def misclosure(l, fx, n_dist):
    w = l - fx;

    w[n_dist:] = (w[n_dist:] + np.pi) % (2 * np.pi) - np.pi;

    return w;

def adjust(l, Cl, dist_pts, ang_pts):
    x = x0.copy();

    for iteration in range(1, 21):
        fx, A_all = design(x, dist_pts, ang_pts);
        w = misclosure(l, fx, len(dist_pts));

        A = A_all[:, free];


        # Weight matrix
        P = np.linalg.inv(Cl);

        # Normal equation matrix
        N = A.T @ P @ A;

        # Normal equation vector
        U = A.T @ P @ w;

        # Solve for corrections to unknown parameters
        delta = np.linalg.inv(N) @ U;
        x[free] += delta;

        if np.max(np.abs(delta)) < 1e-7:
            break;
    
    fx, A_all = design(x, dist_pts, ang_pts);
    r = misclosure(l, fx, len(dist_pts));

    Cx = np.zeros((u,u));

    # Cool notation to loop through all <free> point ids
    Cx[np.ix_(free, free)] = np.linalg.inv(N);

    return x, r, Cx, iteration;

def global_test(r, Cl, u):
    dof = len(r) - u;
    P = np.linalg.inv(Cl);
    chi_stat = float(r @ P @ r);
    sigma0_sq = chi_stat / dof;

    lower = chi2.ppf(0.025, dof);
    upper = chi2.ppf(0.975, dof);

    passed = lower <= chi_stat <= upper

    return chi_stat, sigma0_sq, lower, upper, passed;

l1, Cl1, dist_pts1, ang_pts1 = read_epoch("distances1.txt", "angles1.txt");
l2, Cl2, dist_pts2, ang_pts2 = read_epoch("distances2.txt", "angles2.txt");

# Check to make sure the functions parsed the data properly
assert l1.size == 42 and l2.size == 42;
assert Cl1. shape == (42, 42) and Cl2.shape == (42, 42);

xhat1, r1, Cx1, iter1 = adjust(l1, Cl1, dist_pts1, ang_pts1);
xhat2, r2, Cx2, iter2 = adjust(l2, Cl2, dist_pts2, ang_pts2);

for epoch, xhat, r, Cx, n_dist, iterations in ((1, xhat1, r1, Cx1, len(dist_pts1), iter1), (2, xhat2, r2, Cx2, len(dist_pts2), iter2)):
    print(f"\nEpoch {epoch}");
    print("Iterations:", iterations);
    print("Number of observations:", len(r));
    print("Degrees of freedom:", len(r) - len(free));
    print("Adjusted coordinates (m):\n", xhat.reshape(-1, 2));
    print("Standard deviations (mm):\n", np.sqrt(np.diag(Cx)).reshape(-1, 2) * 1000);
    print("Covariance matrix (mm^2), order x1 y1 ... x5 y5:\n", np.round(Cx * 1e6, 4));
    print("Distance residuals (mm), one row per distance, columns = rounds:\n",
          np.round(r[:n_dist].reshape(-1, 3) * 1000, 2));
    print("Angle residuals (arcsec), one row per angle, columns = rounds:\n",
          np.round(r[n_dist:].reshape(-1, 3) * RHO, 2));

    np.savetxt(f"epoch{epoch}_xhat.txt", xhat.reshape(-1, 2), fmt="%.8f",
               header="X(m) Y(m)");
    np.savetxt(f"epoch{epoch}_Cxhat.txt", Cx, fmt="%.12e",
               header="Covariance (m^2), order x1 y1 x2 y2 x3 y3 x4 y4 x5 y5");

# Task 2 
T1, s2_1, low1, high1, passed1 = global_test(r1, Cl1, len(free));
T2, s2_2, low2, high2, passed2 = global_test(r2, Cl2, len(free));
dof1 = len(r1) - len(free);
dof2 = len(r2) - len(free);

print("\nTask 2a");
for epoch, T, s2, low, high, passed in ((1, T1, s2_1, low1, high1, passed1), (2, T2, s2_2, low2, high2, passed2)):
    print(f"Epoch {epoch} chi-square: {T:.6f}");
    print(f"Epoch {epoch} variance factor: {s2:.6f}");
    print(f"Epoch {epoch} acceptance interval: [{low:.6f}, {high:.6f}]");
    print(f"Epoch {epoch} test:", "PASS" if passed else "FAIL");

if s2_1 >= s2_2:
    F_bartlett, dof_top, dof_bottom = s2_1 / s2_2, dof1, dof2;
else:
    F_bartlett, dof_top, dof_bottom = s2_2 / s2_1, dof2, dof1;
F_bartlett_critical = f.ppf(0.95, dof_top, dof_bottom);

print("\nTask 2b")
print(f"Bartlett test statistic: {F_bartlett:.6f}")
print(f"Critical value F(0.95; {dof_top}, {dof_bottom}): {F_bartlett_critical:.6f}")
print("Bartlett test:", "PASS" if F_bartlett <= F_bartlett_critical else "FAIL")



# Task 3) GCT 
d = xhat2 - xhat1;
Q_d = Cx1 + Cx2;
r_d = np.linalg.matrix_rank(Q_d);
r_c = dof1 + dof2;
s2_c = (T1 + T2) / r_c;

Q_d_inv = np.linalg.pinv(Q_d);
phi_d = d.T @ Q_d_inv @ d;
phi_d = float(phi_d);

y = phi_d / (r_d * s2_c);
y_crit = f.ppf(0.95, r_d, r_c);


print("\nTask 3")
print("Deformation vector (mm), rows = points 1 to 5, columns = dx dy:\n",
      np.round(d.reshape(-1, 2) * 1000, 2))
print(f"Combined variance factor: {s2_c:.6f}")
print(f"Quadratic form d'Qd+d: {phi_d:.6f}")
print("Rank of Qd:", r_d)
print(f"Congruency test statistic: {y:.6f}")
print(f"Critical value F(0.95; {r_d}, {r_c}): {y_crit:.6f}")
print("Global congruency test:", "PASS" if y <= y_crit else "FAIL")
