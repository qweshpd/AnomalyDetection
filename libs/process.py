#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Basic data processing library. Numpy dependent.
'''

import numpy as np

def detect_cusum(x, threshold, drift, ending):
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
        index of when the change ended (if 'ending' is True).
    amp : 1D array_like, float
        amplitude of changes (if 'ending' is True).
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

def _plotcusum(x, threshold, drift, ending, ta, tai, taf, gp, gn):
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

    
def detect_peaks(x, mph = None, mpd = 1, threshold = 0, edge='rising'):

    """Detect peaks in data based on their amplitude and other features.

    Parameters
    ----------
    x : 1D array_like
        data.
    mph : {None, number}, optional (default = None)
        detect peaks that are greater than minimum peak height.
    mpd : positive integer, optional (default = 1)
        detect peaks that are at least separated by minimum peak distance (in
        number of data).
    threshold : positive number, optional (default = 0)
        detect peaks (valleys) that are greater (smaller) than 'threshold'
        in relation to their immediate neighbors.
    edge : {None, 'rising', 'falling', 'both'}, optional (default = 'rising')
        for a flat peak, keep only the rising edge ('rising'), only the
        falling edge ('falling'), both edges ('both'), or don't detect a
        flat peak (None).
    valley : bool, optional (default = False)
        if True (1), detect valleys (local minima) instead of peaks.
    show : bool, optional (default = False)
        if True (1), plot data in matplotlib figure.

    Returns
    -------
    ind : 1D array_like
        indeces of the peaks in 'x'.

    Notes
    -----
    The detection of valleys instead of peaks is performed internally by simply
    negating the data: `ind_valleys = detect_peaks(-x)`
    
    The function can handle NaN's 

    """

    x = np.atleast_1d(x).astype('float64')
    if x.size < 3:
        return np.array([], dtype=int)

    # find indexes of all peaks
    dx = x[1:] - x[:-1]
    # handle NaN's
    indnan = np.where(np.isnan(x))[0]
    if indnan.size:
        x[indnan] = np.inf
        dx[np.where(np.isnan(dx))[0]] = np.inf
    ine, ire, ife = np.array([[], [], []], dtype = int)
    if not edge:
        ine = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) > 0))[0]
    else:
        if edge.lower() in ['rising', 'both']:
            ire = np.where((np.hstack((dx, 0)) <= 0) & (np.hstack((0, dx)) > 0))[0]
        if edge.lower() in ['falling', 'both']:
            ife = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) >= 0))[0]
    ind = np.unique(np.hstack((ine, ire, ife)))
    # handle NaN's
    if ind.size and indnan.size:
        # NaN's and values close to NaN's cannot be peaks
        ind = ind[np.in1d(ind, np.unique(np.hstack((indnan, indnan-1, indnan+1))), invert=True)]
    # first and last values of x cannot be peaks
    if ind.size and ind[0] == 0:
        ind = ind[1:]
    if ind.size and ind[-1] == x.size-1:
        ind = ind[:-1]
    # remove peaks < minimum peak height
    if ind.size and mph is not None:
        ind = ind[x[ind] >= mph]
    # remove peaks - neighbors < threshold
    if ind.size and threshold > 0:
        dx = np.min(np.vstack([x[ind]-x[ind-1], x[ind]-x[ind+1]]), axis=0)
        ind = np.delete(ind, np.where(dx < threshold)[0])
    # detect small peaks closer than minimum peak distance
    if ind.size and mpd > 1:
        ind = ind[np.argsort(x[ind])][::-1]  # sort ind by peak height
        idel = np.zeros(ind.size, dtype=bool)
        for i in range(ind.size):
            if not idel[i]:
                # keep peaks with the same height if kpsh is True
                idel = idel | (ind >= ind[i] - mpd) & (ind <= ind[i] + mpd) \
                    & (x[ind[i]] > x[ind])# if kpsh else True)
                idel[i] = 0  # Keep current peak
        # remove the small peaks and sort back the indexes by their occurrence
        ind = np.sort(ind[~idel])

    return ind
   

def detect_onset(x, threshold = 0, n_above = 1, n_below = 0,
                 threshold2 = None, n_above2 = 1):
    
    """Detects onset in data based on amplitude threshold.

    Parameters
    ----------
    x : 1D array_like
        data.
    threshold : number, optional (default = 0)
        minimum amplitude of `x` to detect.
    n_above : number, optional (default = 1)
        minimum number of continuous samples >= `threshold`
        to detect (but see the parameter `n_below`).
    n_below : number, optional (default = 0)
        minimum number of continuous samples below `threshold` that
        will be ignored in the detection of `x` >= `threshold`.
    threshold2 : number or None, optional (default = None)
        minimum amplitude of `n_above2` values in `x` to detect.
    n_above2 : number, optional (default = 1)
        minimum number of samples >= `threshold2` to detect.
    Returns
    -------
    inds : 2D array_like [indi, indf]
        initial and final indeces of the onset events.
"""

    x = np.atleast_1d(x).astype('float64')
    # deal with NaN's (by definition, NaN's are not greater than threshold)
    x[np.isnan(x)] = -np.inf
    # indices of data greater than or equal to threshold
    inds = np.nonzero(x >= threshold)[0]
    if inds.size:
        # initial and final indexes of almost continuous data
        inds = np.vstack((inds[np.diff(np.hstack((-np.inf, inds))) > n_below+1], \
                          inds[np.diff(np.hstack((inds, np.inf))) > n_below+1])).T
        # indexes of almost continuous data longer than or equal to n_above
        inds = inds[inds[:, 1]-inds[:, 0] >= n_above-1, :]
        # minimum amplitude of n_above2 values in x to detect
        if threshold2 is not None and inds.size:
            idel = np.ones(inds.shape[0], dtype=bool)
            for i in range(inds.shape[0]):
                if np.count_nonzero(x[inds[i, 0]: inds[i, 1]+1] >= threshold2) < n_above2:
                    idel[i] = False
            inds = inds[idel, :]
    if not inds.size:
        inds = np.array([])  # standardize inds shape for output
    return inds
