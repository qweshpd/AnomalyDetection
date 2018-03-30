"""Cumulative sum algorithm (CUSUM) to detect abrupt changes in data."""

from __future__ import division, print_function
import numpy as np

def detect_cusum(x, threshold, drift, ending, show):
    '''
    Cumulative sum algorithm (CUSUM) to detect abrupt changes in data.

    Parameters
    ----------
    x : 1D array_like
        data.
    threshold : positive number, optional (default = 1)
        amplitude threshold for the change in the data.
    drift : positive number, optional (default = 0)
        drift term that prevents any change in the absence of change.
    ending : bool, optional (default = False)
        True (1) to estimate when the change ends; False (0) otherwise.
    show : bool, optional (default = True)
        True (1) plots data in matplotlib figure, False (0) don't plot.

    Returns
    -------
    ta : 1D array_like [indi, indf], int
        alarm time (index of when the change was detected).
    tai : 1D array_like, int
        index of when the change started.
    taf : 1D array_like, int
        index of when the change ended (if `ending` is True).
    amp : 1D array_like, float
        amplitude of changes (if `ending` is True).
    '''

    x = np.atleast_1d(x).astype('float64')
    gp, gn = np.zeros(x.size), np.zeros(x.size)
    ta, tai, taf = np.array([[], [], []], dtype=int)
    tap, tan = 0, 0
    amp = np.array([])
    # Find changes (online form)
    for i in range(1, x.size):
        s = x[i] - x[i - 1]
        if gp[i - 1] > threshold or gn[i - 1] > threshold:  # change detected!
            ta = np.append(ta, i - 1)    # alarm index
            tai = np.append(tai, tap if gp[i - 1] > threshold else tan)  # start
            # reset alarm
            gp[i] = 0 + s - drift
            gn[i] = 0 - s - drift   
        else:
            gp[i] = gp[i - 1] + s - drift  # cumulative sum for + change
            gn[i] = gn[i - 1] - s - drift  # cumulative sum for - change
        
        if gp[i] < 0:
            gp[i], tap = 0, i
        if gn[i] < 0:
            gn[i], tan = 0, i
        
    # THE CLASSICAL CUSUM ALGORITHM ENDS HERE

    # Estimation of when the change ends (offline form)
    if tai.size and ending:
        _, tai2, _, _, _, _ = detect_cusum(x[::-1], threshold, drift, False, False)
        taf = x.size - tai2[::-1] - 1
        # Eliminate repeated changes, changes that have the same beginning
        tai, ind = np.unique(tai, return_index=True)
        ta = ta[ind]
        # taf = np.unique(taf, return_index=False)  # corect later
        if tai.size != taf.size:
            if tai.size < taf.size:
                taf = taf[[np.argmax(taf >= i) for i in ta]]
            else:
                ind = [np.argmax(i >= ta[::-1])-1 for i in taf]
                ta = ta[ind]
                tai = tai[ind]
        # Delete intercalated changes (the ending of the change is after
        # the beginning of the next change)
        ind = taf[:-1] - tai[1:] > 0
        if ind.any():
            ta = ta[~np.append(False, ind)]
            tai = tai[~np.append(False, ind)]
            taf = taf[~np.append(ind, False)]
        # Amplitude of changes
        amp = x[taf] - x[tai]

    return ta, tai, taf, amp, gp, gn


def _plot(x, threshold, drift, ending, ta, tai, taf, gp, gn):
    '''Plot results of the detect_cusum function.'''

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib is not available.')
    else:
        _, (ax1, ax2) = plt.subplots(2, 1, figsize = (15, 6))

        t = range(x.size)
        ax1.plot(t, x, 'b-', lw=2)
        if len(ta):
            ax1.plot(tai, x[tai], '>', mfc = 'g', mec = 'g', ms = 10,
                     label='Start')
            if ending:
                ax1.plot(taf, x[taf], '<', mfc='g', mec='g', ms=10,
                         label = 'Ending')
            ax1.plot(ta, x[ta], 'o', mfc = 'r', mec = 'r', mew = 1, ms = 5,
                     label = 'Alarm')
            ax1.legend(loc = 'best', framealpha = .5, numpoints = 1)
#        ax1.set_xlim(-.01*x.size, x.size*1.01-1)
#        ax1.set_xlabel('Data #', fontsize=14)
#        ax1.set_ylabel('Amplitude', fontsize=14)
#        ymin, ymax = x[np.isfinite(x)].min(), x[np.isfinite(x)].max()
#        yrange = ymax - ymin if ymax > ymin else 1
#        ax1.set_ylim(ymin - 0.1*yrange, ymax + 0.1*yrange)
#        ax1.set_title('Time series and detected changes ' +
#                      '(threshold= %.3g, drift= %.3g): N changes = %d'
#                      % (threshold, drift, len(tai)))
        ax2.plot(t, gp, 'y-', label = '+')
        ax2.plot(t, gn, 'm-', label = '-')
#        ax2.set_xlim(-.01*x.size, x.size*1.01-1)
#        ax2.set_xlabel('Data #', fontsize=14)
#        ax2.set_ylim(-0.01*threshold, 1.1*threshold)
        ax2.axhline(threshold, color = 'r')
#        ax1.set_ylabel('Amplitude', fontsize=14)
#        ax2.set_title('Time series of the cumulative sums of ' +
#                      'positive and negative changes')
        ax2.legend(loc = 'best', framealpha = .5, numpoints = 1)
        plt.tight_layout()
        plt.show()
