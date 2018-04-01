# Kalman filter example demo in Python

# A Python implementation of the example given in pages 11-15 of "An
# Introduction to the Kalman Filter" by Greg Welch and Gary Bishop,
# University of North Carolina at Chapel Hill, Department of Computer
# Science, TR 95-041,
# http://www.cs.unc.edu/~welch/kalman/kalmanIntro.html

# by Andrew D. Straw
import psutil
import numpy as np
import matplotlib.pyplot as plt


plt.rcParams['figure.figsize'] = (10, 8)
plt.ion()

# intial parameters
interval = 1 # sleep time
n_iter = 300 # total runtime
threshold = 60.0
sz = (n_iter,) # size of array
z = []  # observations 

Q = 1e-2 # process variance

# allocate space for arrays
xhat = []      # a posteri estimate of x
P = np.zeros(sz)         # a posteri error estimate
xhatminus = np.zeros(sz) # a priori estimate of x
Pminus = np.zeros(sz)    # a priori error estimate
K = np.zeros(sz)         # gain or blending factor

R = 0.1**2 # estimate of measurement variance, change to see effect

# intial guesses
xhat.append(psutil.cpu_percent(0))
P[0] = 1.0
z.append(psutil.cpu_percent(0))

plt.figure()
plt.plot(z[0], 'b--', label = 'Measure')
plt.plot(xhat[0], 'r-', label = 'KalmanFilter')
plt.legend(loc = 2)

for k in range(1, n_iter):
    z.append(psutil.cpu_percent(0))
    # time update
    xhatminus[k] = xhat[k - 1]
    Pminus[k] = P[k - 1] + Q

    # measurement update
    K[k] = Pminus[k]/( Pminus[k] + R )
    xhat.append(xhatminus[k] + K[k]*(z[k] - xhatminus[k]))
    P[k] = (1 - K[k]) * Pminus[k]
    
    if xhat[k] >= max(1.05 * xhat[k - 1], float(threshold)):
        plt.plot([k - 1, k], z[-2 :], 'b--')
    else:
        plt.plot([k - 1, k], z[-2 :], 'y--')
    
    # plt.plot([k - 1, k], z[k - 1 :], 'b--')
    plt.plot(xhat, 'r-')
    plt.pause(interval)




    

