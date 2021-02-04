#!/usr/bin/env python

# Import neccessary Modules
import numpy as np
from numpy import cos, pi
import scipy.signal as sg
from scipy.stats import chi2, norm
from scipy.special import gammaincinv
import math, copy
from sys import exit
import pdb

### Gererate time series from a PSD spectrum
def genTimeSeriesFromPSD(S,fs):
    """ genTimeSeriesFromPSD (S,fs)

    A simple function to generate time series from a given *one-sided* power spectrum
    via iFFT. It draws the amplitudes from a chi2 distribution, while it asigns
    a random phase between [-2*pi, 2*pi].

    NK 2019
    """

    N    = 2*(len(S)-1)
    Ak   = np.sqrt(N*S*fs)               # Take the sqrt of the amplitude
    Ak   = np.append(Ak, Ak[1:-1][::-1]) # make two-sided
    rphi = np.zeros(N)

    rphi[1:int(N/2)-1]   = 2*np.pi*np.random.uniform(0, 1, int(N/2)-2)  # First half
    rphi[int(N/2)]       = 2*np.pi*round(np.random.uniform(0, 1))       # mid point
    rphi[int(N/2+1):N-1] = -rphi[1:int(N/2)-1][::-1]                    # reflected half

    X  = Ak*np.sqrt(chi2.rvs(2, size=N)/2) * np.exp(1j * rphi)

    return np.fft.ifft(X) # return Inverse FFT

# Generic window function
def window(winType, N, alpha=0.01):
    """ window (WINTYPE, N, alpha=0.01)

    A function to generate a desired type of window, given the data length N.
    The 'alpha' parameter denotes the additional smoothness parameter neccesssary
    for some particular types of windows, i.e. Tukey, Planck, or Kaizer.

    Available window types:

        nuttall, nuttall3, nuttall3a, nuttall3b, nuttall4, nuttall4a, nuttall4b,
        nuttall4c, blackman-harris, tukey, planck, welch, rectangular, hamming,
        hanning, kaiser, sft3f, sft5f, sft5m, ftni, fthp, ftsrs, hft70, hft90d,
        hft95, hft116d, hft144d, hft169d, hft196d, hft223d, hft248d

    For more details about the window types check http://hdl.handle.net/11858/00-001M-0000-0013-557A-5

    INPUTS: a) WINTYPE: String of the desired window.
            b) N:       The length of the window.
            c) alpha:   Smoothness parameter.

    RETURN:
            a) WINVALS:     The window values.
            b) WINVALSNROM: The window values normalized to the area of the window.

    NK 2019
    """

    # Define the more common z
    z       = (np.arange(0,N,1)/N) * 2.0 * pi
    N       = int(N)
    winType = winType.lower() # Be a little more robust on the inputs

    # Tukey window
    def tukey(N, alpha):
        # alpha -- parameter the defines the shape of the window
        w         = np.zeros(N, dtype=float)
        i         = np.arange(0,N,1)
        r         = (2.0*i)/(alpha*(N-1))
        l1        = int(np.rint((alpha*(N-1))/2.0))
        l2        = int(np.rint((N-1)*(1.0-alpha/2.0)))
        w[0:l1]   = 0.5*(1.0 + cos(pi*(r[0:l1] - 1.0)))
        w[l1:l2]  = 1.0
        w[l2:N-1] = 0.5*(1+cos(pi*(r[l2:N-1] - 2.0/alpha + 1.0)))
        return w

    # planck window
    def planck(N, epsilon):
        # alpha -- parameter the defines the shape of the window
        w         = np.zeros(N, dtype=float)
        i         = np.arange(0,N,1)
        Z_plus    = 2.*epsilon*( 1./(1. + ((2.*i)/(N-1) - 1)) + 1./(1. - 2*epsilon + ((2.*i)/(N-1) - 1)) )
        Z_minus   = 2.*epsilon*( 1./(1. - ((2.*i)/(N-1) - 1)) + 1./(1. - 2*epsilon - ((2.*i)/(N-1) - 1)) )
        l1        = int(np.rint( epsilon * (N-1) ))
        l2        = int(np.rint( (1-epsilon) * (N-1) ))
        w[0:l1]   = 1./(np.exp(Z_plus[0:l1]) + 1)
        w[l1:l2]  = 1.0
        w[l2:N-1] = 1./(np.exp(Z_minus[l2:N-1]) + 1)
        return w

    # fthp
    def fthp(z):
        return (1 + 1.912510941 * cos (z) + \
                       1.079173272 * cos (2 * z) + \
                       0.1832630879 * cos (3 * z))

    # ftsrs
    def ftsrs(z):
        return (1.0 - 1.93 * cos (z) + \
                       1.29 * cos (2 * z) - 0.388 * cos (3 * z) + 0.028 * cos (4 * z))

    # ftni
    def ftni(z):
        return 0.2810639 - 0.5208972 * cos (z) + 0.1980399 * cos (2 * z)

    # hft70
    def hft70(z):
        return (1 - 1.9575375 * cos (z) + \
                      1.4780705 * cos (2 * z) - 0.6367431 * cos (3 * z) + \
                      0.1228389 * cos (4 * z) - 0.0066288 * cos (5 * z))

    # hft90d
    def hft90d(z):
        return (1 - 1.942604 * cos (z) + \
                       1.340318 * cos (2 * z) - 0.440811 * cos (3 * z) + \
                       0.043097 * cos (4 * z))

    # hft95
    def hft95(z):
        return (1 - 1.9383379 * cos (z) + \
                      1.3045202 * cos (2 * z) - 0.4028270 * cos (3 * z) + \
                      0.0350665 * cos (4 * z))

    # hft116d
    def hft116d(z):
        return (1 - 1.90796 * cos (z) + \
                       1.07349 * cos (2 * z) - 0.18199 * cos (3 * z))

    # hft144d
    def hft144d(z):
        return (1 - 1.9575375 * cos (z) + \
                      1.4780705 * cos (2 * z) - 0.6367431 * cos (3 * z) + \
                      0.1228389 * cos (4 * z) - 0.0066288 * cos (5 * z))

    # hft169d
    def hft169d(z):
        return (1 - 1.97441843 * cos (z) \
                      + 1.65409889 * cos (2 * z) - 0.95788187 * cos (3 * z) + \
                      0.33673420 * cos (4 * z) - 0.06364622 * cos (5 * z) + \
                      0.00521942 * cos (6 * z) - 0.00010599 * cos (7 * z))

    # hft196d
    def hft196d(z):
        return (1 - 1.979280420 * cos (z) + \
                      1.710288951 * cos (2 * z) - 1.081629853 * cos (3 * z) + \
                      0.448734314 * cos (4 * z) - 0.112376628 * cos (5 * z) + \
                      0.015122992 * cos (6 * z) - 0.000871252 * cos (7 * z) + \
                      0.000011896 * cos (8 * z))

    # hft223d
    def hft223d(z):
        return (1 - 1.98298997309 * cos(z) + \
                      1.75556083063 * cos (2 * z) - 1.19037717712 * cos (3 * z) + \
                      0.56155440797 * cos (4 * z) - 0.17296769663 * cos (5 * z) + \
                      0.03233247087 * cos (6 * z) - 0.00324954578 * cos (7 * z) + \
                      0.00013801040 * cos (8 * z) - 0.00000132725 * cos (9 * z))

    # hft248d
    def hft248d(z):
        return (1 - 1.985844164102 * cos(z) + \
                      1.791176438506 * cos (2 * z) - 1.282075284005 * cos (3 * z) + \
                      0.667777530266 * cos (4 * z) - 0.240160796576 * cos (5 * z) + \
                      0.056656381764 * cos (6 * z) - 0.008134974479 * cos (7 * z) + \
                      0.000624544650 * cos (8 * z) - 0.000019808998 * cos (9 * z) + \
                      0.000000132974 * cos (10 * z))

    # sft3f
    def sft3f(z):
        return 0.26526 - 0.5 * cos (z) + 0.23474 * cos (2 * z)

    # sft3m
    def sft3m(z):
        return 0.28235 - 0.52105 * cos (z) + 0.19659 * cos (2 * z)

    # sft4f
    def sft4f(z):
        return 0.21706 - 0.42103 * cos (z) + 0.28294 * cos (2 * z) - 0.07897 * cos (3 * z)

    # sft4m
    def sft4m(z):
        return 0.241906 - 0.460841 * cos(z) \
                   + 0.255381 * cos (2 * z) - 0.041872 * cos (3 * z)

    # sft5f
    def sft5f(z):
        return (0.1881 - 0.36923 * cos (z) + \
                      0.28702 * cos (2 * z) - 0.13077 * cos (3 * z) + 0.02488 * cos (4 * z))

    # sft5m
    def sft5m(z):
        return (0.209671 - 0.407331 * cos(z) + \
                      0.281225 * cos (2 * z) - 0.092669 * cos (3 * z) + \
                      0.0091036 * cos (4 * z))

    # kaiser
    def kaiser(N, alpha):
        w  = sg.kaiser(N, beta=alpha)
        return w

    # nuttall
    def nuttall(N, z, type = 'nuttal4'):

        if type == 'nuttall':
            w = sg.get_window('nuttall', N)
        if type == 'nuttall3':
            w = 0.375 - 0.5 * cos (z) + 0.125 * cos (2 * z)
        if type == 'nuttall3a':
            w = 0.40897 - 0.5 * cos (z) + 0.09103 * cos (2 * z)
        elif type == 'nuttall3b':
            w = 0.4243801 - 0.4973406 * cos (z) + 0.0782793 * cos (2 * z)
        elif type == 'nuttall4':
            w = 0.3125 - 0.46875 * cos(z) + 0.1875 * cos (2 * z) - 0.03125 * cos (3 * z)
        elif type == 'nuttall4a':
            w = 0.338946 - 0.481973 * cos (z) + 0.161054 * cos (2 * z) - 0.018027 * cos (3 * z)
        elif type == 'nuttall4b':
            w = 0.355768 - 0.487396 * cos (z) + 0.144232 * cos (2 * z) - 0.012604 * cos (3 * z)
        elif type == 'nuttall4c':
            w = 0.3635819 - 0.4891775 * cos (z) + 0.1365995 * cos (2 * z) - 0.0106411 * cos (3 * z)
        return w

    # welch
    def welch(z):
        w = 1 - (2 * z - 1)**2
        return w

    # Blackman-Harris window
    def bh(z):
        w = 0.35875 - 0.48829 * cos (z) + 0.14128 * cos (2 * z) - 0.01168 * cos (3 * z)
        return w

    # Rectangular window
    def rectangular(N):
        w = sg.boxcar(N)
        return w

    # Hamming window
    def hamm(N):
        w = np.hamming(N)
        return w

    # Hanning window
    def hann(N):
        w = np.hanning(N)
        return w

    # Choose the window
    if winType == 'blackman-harris' or winType == 'bh92':
        winvals = bh(z)
    elif winType == 'tukey':
        winvals = tukey(N,alpha)
    elif winType == 'planck':
        winvals = planck(N, alpha)
    elif winType == 'hamming':
        winvals = hamm(N)
    elif winType == 'hanning':
        winvals = hann(N)
    elif winType == 'welch':
        winvals = welch(z)
    elif winType == 'nuttall':
        winvals = nuttall(N,z,type = 'nuttall')
    elif winType == 'nuttall3':
        winvals = nuttall(N,z,type = 'nuttall3')
    elif winType == 'nuttall3a':
        winvals = nuttall(N,z,type = 'nuttall3a')
    elif winType == 'nuttall3b':
        winvals = nuttall(N,z,type = 'nuttall3b')
    elif winType == 'nuttall4':
        winvals = nuttall(N,z, type = 'nuttall4')
    elif winType == 'nuttall4a':
        winvals = nuttall(N,z, type = 'nuttall4a')
    elif winType == 'nuttall4b':
        winvals = nuttall(N,z, type = 'nuttall4b')
    elif winType == 'nuttall4c':
        winvals = nuttall(N,z, type = 'nuttall4c')
    elif winType == 'rectangular':
        winvals = rectangular(N)
    elif winType == 'kaiser':
        winvals = kaiser(N, alpha=alpha)
    elif winType == 'sft3f':
        winvals = sft3f(z)
    elif winType == 'sft5f':
        winvals = sft5f(z)
    elif winType == 'sft5m':
        winvals = sft5m(z)
    elif winType == 'ftni':
        winvals = ftni(z)
    elif winType == 'fthp':
        winvals = fthp(z)
    elif winType == 'ftsrs':
        winvals = ftsrs(z)
    elif winType == 'hft70':
        winvals = hft70(z)
    elif winType == 'hft90d':
        winvals = hft90d(z)
    elif winType == 'hft95':
        winvals = hft95(z)
    elif winType == 'hft116d':
        winvals = hft116d(z)
    elif winType == 'hft144d':
        winvals = hft144d(z)
    elif winType == 'hft169d':
        winvals = hft169d(z)
    elif winType == 'hft196d':
        winvals = hft196d(z)
    elif winType == 'hft223d':
        winvals = hft223d(z)
    elif winType == 'hft248d':
        winvals = hft248d(z)
    else:
        exit('### Error: unknown window type ... ')

    K            = np.sum(winvals*winvals)
    winvals_norm = winvals/(np.sqrt(K)) # Normalise the window

    return winvals, winvals_norm

# A PSD function, imported from LTPDA.
def psd(x, fs, y= None, navs=-1, nperseg=-1, win='nuttall4b', olap=0, flims=-1, winalpha=0.1, detrend=None, method=None):
    """ welchpsd (data, fs, navs, win='nuttall4b', olap=0, winalpha=0.1)

        A PSD function that returns the one sided power spectrum. Possible to perform
        averaging either by defining the number of averages throught the NAVS option
        of the NFFT number of points per segment.

    INPUTS: a) X:        The data (1D, time series)
            b) fs:       The sampling frequency.
            c) y:        Second time series data to perform the cross-psectral density instead. Default=None.
            d) navs:     The number of averages. Will be always favored if 'nperseg' is also used as an input.
            e) nperseg:  The number of points per segment. 'Navs' will be favored if both are used.
            f) win:      The window type. Can be either string (see window function)
                         or numerical values directly assumed to have the correct length.
            g) olap:     The amount of overlap (values between [0,1]).
            h) flims:    The frequency limits of the putput spectrum. By default the power at all frequencies is exported.
            i) winalpha: The 'alpha' smoothing parameter of the window.
            j) method:   The method to perform the averages. Either running mean, or running median.

    RETURNS: The fvec frequency vector, and the actual spectrum S.

    NK 2019
    """

    # A subunction to loop over the data segmets and perform the average of the FFTs
    def averagefft(X, w, navs, segmentStarts, segmentEnds, nfft, fs, detrend, Y=None, method=None):
        S = []          # Initialize
        C = np.sum(w*w) # Window normalization
        # Get each segment
        for kk in range(navs):
            XX = X[int(segmentStarts[kk]):int(segmentEnds[kk])]
            # Detrend level 0?
            if detrend is not None:
                if detrend == 1: XX = XX - np.mean(XX)
            # Do the FFTs here
            Xw = XX*w
            Xf = np.fft.fft(Xw, n=int(nfft))

            # Check second input?
            if Y is not None:
                YY = Y[int(segmentStarts[kk]):int(segmentEnds[kk])]
                # Do the FFTs here
                Yw = YY*w
                Yf = np.fft.fft(Yw, n=int(nfft))
                # Compute the spectra for the k-th segment and save it into array
                S.append(Xf*np.conjugate(Yf)/C)
            else:
                # Compute the spectra for the k-th segment and save it into array
                S.append(Xf*np.conjugate(Xf)/C)

        Sxx, Vxx = WelfordMean(S) # Do the average (running mean & variance)
        
        if method is not None: # Do the average (for the case of a running median only)
            Sxx = np.median(np.array([S]), axis=1)[0]/0.7023319615912207

        # Compute the errors depending on the number of averages 
        if navs == 1:
            Vxx = []
        else:
            Vxx = Vxx/(navs-1)/navs

        return np.absolute(Sxx), np.absolute(Vxx)

    # A subfunction to scale the averaged periodogram to PSD
    def scale(S_in, Se_in, nfft, fs, flims):
            # Take 1-sided spectrum which means we double the power in the appropriate bins
            if nfft%2 == 0:
                N         = int((nfft+1)/2)
                indices   = range(0, N)     # ODD
                Sxx1sided = S_in[indices]
                S_in      = np.concatenate(( np.array([Sxx1sided[0]], float), 2*Sxx1sided[1:])) # Double the power except for the DC bin
                if len(Se_in)>0:
                    Svxx1sided = Se_in[indices]
                    Se_in      = np.concatenate((np.array([Svxx1sided[0]], float), 4*Svxx1sided[1:]))
            else:
                N         = int(nfft/2+1)
                indices   = range(0, N)     # EVEN
                Sxx1sided = S_in[indices]
                S_in      = np.concatenate((np.array([Sxx1sided[0]], float), \
                                            2*Sxx1sided[1:-1], np.array([Sxx1sided[-1]], float))) # Double power except the DC bin and the Nyquist bin
                if len(Se_in)>0:
                    Svxx1sided = Se_in[indices] # Take only [0,pi] or [0,pi)
                    Se_in      = np.concatenate((np.array([Svxx1sided[0]], float), 4*Svxx1sided[1:-1], np.array([Svxx1sided[-1]], float)))

            # Now scale to PSD
            S  = np.array(S_in) / fs
            if len(Se_in)>0:
                Se = np.array(Se_in) / (fs**2)
            else:
                Se = Se_in

            # Make the positive frequency vector
            f = np.fft.fftfreq(int(nfft), d=1/fs)[:N]

            # Chop in frequency?
            if flims != -1 and len(flims) == 2:
                ind = np.where(np.logical_and(f>=flims[0], f<=flims[1]))
                f   = f[ind]
                S   = S[ind]
                if len(Se_in)>0:
                    Se  = Se[ind]

            return f, S, Se

    # Get segments length
    L = obj_len = len(x)

    # Check second data input
    if y is not None:
        if len(y) != L:
            exit('\t >>> Length of time-series is not the same. Aborting.')

    # Check the input values for NFFT and NAVS
    if  (navs == -1 and nperseg == -1):
        navs = 1
    elif (navs != -1 and nperseg == -1):
        navs = np.absolute(navs)
    else:
        if nperseg > len(x):
            nperseg = len(x)
        nperseg = np.absolute(nperseg)
        xOlap   = np.round(olap*nperseg)
        navs    = int(np.fix((L - xOlap) / (nperseg - xOlap)))

    # Compute the number of segments
    L = np.round(obj_len/(navs*(1-olap) + olap))

    # Checks it will really obtain the correct answer.
    # This is needed to cope with the need to work with integers
    while True:
        if np.fix((obj_len-np.round(L*olap))/(L-np.round(L*olap))) < navs:
            L = L - 1
        else:
            break

    nfft = L # This is the calculated NFFT now

    # Get the window
    if type(win) == str:
        w, _ = window(win, L, alpha = winalpha)
    else:
        print('\t >>> Array of numerical values for the window was probably inserted. I will try to compute the PSD assuming the window has the correct length.')
        w = win

    # Compute start and end indices of each segment
    xOlap         = np.round(olap*nfft)
    segmentStep   = nfft-xOlap
    segmentStarts = range(0, int(navs*segmentStep), int(segmentStep))
    segmentEnds   = segmentStarts+nfft

    # Perform the actual PSD calculation
    S, Se = averagefft(x, w, navs, segmentStarts, segmentEnds, nfft, fs, detrend, Y=y, method=method)

    # Scale to PSD
    f, S, Se = scale(S, Se, nfft, fs, flims)

    return f, S, Se



# Wrapper around the scipy.signal Welch PSD function.
def welchpsd(data, fs, nperseg, win='nuttall4b', olap=0, winalpha=0.1, detrend=False, fmin=None, fmax=None):
    """ welchpsd (data, fs, nperseg, win='nuttall4b', olap=0, winalpha=0.1)

        A simple function/wrapper around the scipy.signal Welch.

    INPUTS: a) data:     The data (1D, time series)
            b) fs:       The sampling frequency.
            c) nperseg:  The number of data points per segment to perform the averages.
            d) win:      The window type.
            e) olap:     The amount of overlap (values between [0,1]).
            f) winalpha: The 'alpha' smoothing parameter of the window.
            g) detrend:  The detrend order before computing the PSD (see scipy.signal.welch)
            h) fmin:     The minimum frequency of the analysis
            i) fmax:     The maximum frequency of the analysis

    RETURNS: The fvec frequency vector, and the actual spectrum S.

    NK 2019
    """
    # Get the window
    if type(win) == str:
        w, _ = window(win, nperseg, alpha = winalpha)
    else:
        print('\t A numerical value for the window was inserted. I will try to compute the PSD assuming the window has the correct length.')
        w = win

    f, S = sg.welch(data,
                    fs=fs,
                    window=w,
                    nperseg=nperseg,
                    detrend=detrend,
                    return_onesided=True,
                    noverlap=int(nperseg * olap))

    # Clip the spectra with respect to flims specified:
    if fmin is None:
        fmin = f[0]
    if fmax is None:
        fmax = f[-1]
    ind  = np.where(np.logical_and(f>=fmin, f<=fmax))
    return f[ind], S[ind]

# Wrapper around the scipy.signal CPSD function
def cpsd(X, Y, fs, nperseg, win='nuttall4b', olap=0, winalpha=0.1, onesided=True):
    """ cpsd (X, Y, fs, nperseg, win='nuttall4b', olap=0, winalpha=0.1)

        A simple function/wrapper around the scipy.signal cpsd.

    INPUTS: a) X:        The first data input (1D, time series).
            b) Y:        The second data input (1D, time series).
            c) fs:       The sampling frequency.
            d) nperseg:  The number of data points per segment to perform the averages.
            e) win:      The window type.
            f) olap:     The amount of overlap (%).
            g) winalpha: The 'alpha' smoothing parameter of the window.

    RETURNS: The fvec frequency vector, and the actual cross-spectrum S.

    NK 2019
    """
    # Get the window
    w, _ = window(win, nperseg, alpha = winalpha)

    # Do the csd
    f, S = sg.csd(X, Y,
                   fs=fs,
                   window=w,
                   nperseg=nperseg,
                   noverlap=int(nperseg * olap),
                   return_onesided=True, scaling='density')
    if onesided:
        return f, S
    else:
        return f, S

# Compute TD
def ComputeTD(FS, dt):
    """ ComputeTD (FS, dt)

        A simple function to IFFT to time domain.

    INPUTS: a) FS: The data (1D, frequency series)
            b) dt: The cadence, Delta-t.

    RETURNS: The time series TS, and tvec, the time vector starting at zero.

    NK 2019
    """
    return np.arange(len(FS))*dt, np.fft.irfft(FS*(1.0/dt))

# BinData
def binData(y, x, resolution=100.0, type='mean'):
    """ binData (X, Y, resolution=100.0, type)

        A simple function to bin frequency series data.

    INPUTS: a) Y: The data (1D)
            a) X: The x-values of the data (1D)
            c) resolution: The resolution
            d) type: Take the mean or the median

    RETURNS: The binned Y, X, and DY data.

    NK 2019
    """
    # Do a simple check
    availableTypes = ['mean', 'median']
    if type not in availableTypes:
        raise ValueError("Invalid type. Expected one of: %s" % availableTypes)

    xmin = x[0]
    xmax = x[-1]
    alph = 10**(1/resolution)

    # number of bins in the rebinned data set
    N = math.ceil(np.log10(xmax/xmin) * resolution)

    # maximum and minimum x-value for each bin
    x_min = xmin*alph**np.array(range(N))
    x_max = xmin*alph**np.array(range(1, N+1))
    dyr   = np.zeros((1,N))
    nr    = np.zeros((1,N))
    yr    = np.zeros((1,N), dtype = np.complex_)
    xr    = np.zeros((1,N))

    for kk in range(N):
        ind = np.argwhere( (x >= x_min[kk]) & (x < x_max[kk]) )
        if np.any(ind):
            nr[0,kk]  = ind.shape[0]
            if type == 'mean':
                xr[0,kk]  = np.mean(x[ind])     # rebinned x bins
                yr[0,kk]  = np.mean(y[ind])     # rebinned y bins
            else:
                xr[0,kk]  = np.median(x[ind])   # rebinned x bins
                yr[0,kk]  = np.median(y[ind])   # rebinned y bins
            dyr[0,kk] = np.std(y[ind])/np.sqrt(nr[0,kk])

    return np.ravel(yr), np.ravel(xr), np.ravel(dyr)

def WelfordMean(y):
    """ WelfordMean (Y)

        The Welford method of computing/updating the mean.

    INPUTS: a) Y: The data. A list containing the segmented data at each position.

    RETURN:
            a) MU:  The mean.
            b) SIG: The error.

    NK 2019
    """
    mu = np.copy(y[0])
    s  = 0.0
    for nn in range(1, len(y)):
        X     = np.copy(y[nn])
        delta = X - mu
        mu    = mu + (delta/(nn+1))
        s     = s + delta * (X - mu)
    return mu, s

def lpsd(d, fs, Kdes=100, Jdes=300, Lmin=0, flims=None, win='nuttall4b', winalpha=0.2, olap=0.0, order=0, errrype='std', DOPLOT=False, VERBOSE=False):
    """ lpsd (d, fs, Kdes=100, Jdes=300, Lmin=0, win='nuttall4b', winalpha=0.2, olap=0.0, order=0, errrype='std', DOPLOT=False, VERBOSE=False)

        Power spectrum estimation on logarithmically spaced frequency grid. 

        Inputs: 

            d: The time series data. A numpy array object, or for multiple channels a list of numpy arrays.
           fs: The sampling frequency.
         Kdes: The minimum number of averages to be used.
         Jdes: The number of frequencies to be computed, qually spaced logaritmically.
         Lmin: The minimum NFFT length to be used. 
        flims: The minimum and maximum frequencies of the analysis [f_min, fmax].
          win: The window.
     winalpha: The alpha parameter calibrates the window.
         olap: The amount in overlap (in %, takes values from [0,100]).
        order: The detrending order for each segment.
      errtype: Two ways of calculating the errors are available. The 'std' and 'analytic'.
       DOPLOT: Fasle-True flag to make a plot of te resulting LPSD.
      VERBOSE: Fasle-True flag to print information during hte computation.

        Outputs: f, S, Se, ENBW: Thr frequency vector, the PSD, the variance Se (min and max), the ENBW

                The outptus are either numpy arrays of lists of numpy arrays, depending on the number of channels.

        This is the Python implementation of the LPSD function of the LTPDA MATLAB toolbox.
        For more details, see https://www.elisascience.org/ltpda/ 

        References:  "Improved spectrum estimation from digitized time series
                      on a logarithmic frequency axis", Michael Troebs, Gerhard Heinzel,
                      Measurement 39 (2006) 120-129.

    NK 2020
    """
    # Helper function to define inputs for the DFT
    def ltf_plan(Ndata, fs, olap, Lmin, Jdes, Kdes, flims):
        xov     = (1 - olap/100)
        fmin    = fs / Ndata 
        fmax    = fs/2
        fresmin = fs / Ndata
        freslim = fresmin * (1+xov*(Kdes-1))
        logfact = (Ndata/2)**(1/Jdes) - 1
        fi      = fmin 
        bmin    = 1
        f, r, b, L, K = ([] for i in range(5)) # Init
        # Loop over frequencies
        while fi < fmax:
            fres = fi * logfact
            if fres <= freslim: fres = np.sqrt(fres*freslim)
            if fres < fresmin: fres = fresmin
            fbin = fi/fres
            if fbin < bmin:
                fbin = bmin
                fres = fi/fbin
            dftlen = np.round(fs / fres)
            if dftlen > Ndata: dftlen = Ndata
            if dftlen < Lmin: dftlen = Lmin
            nseg = np.round((Ndata - dftlen) / (xov*dftlen) + 1)
            if nseg == 1: dftlen = Ndata
            fres = fs / dftlen
            fbin = fi / fres
            # Store outputs
            f.append(fi)
            r.append(fres)
            b.append(fbin)
            L.append(dftlen)
            K.append(int(nseg))
            fi = fi + fres
        ind = range(len(f))
        if flims is not None: # split in frequencies
            ind = np.where(np.logical_and(np.array(f)>=flims[0], np.array(f)<=flims[1])) 
        return np.array(f)[ind], np.array(r)[ind], np.array(b)[ind], np.array(L)[ind], np.array(K)[ind]
    # Detrend function
    def detrend(X, order):
        if order == -1:
            Y = copy.deepcopy(X)
        elif order == 0:
            Y = X - np.mean(X)
        elif order >= 1:
            exit('\t >>> ERROR: I can detrend up to order of 0 for now. Sorry ... ')
        return Y
    # rounding function (Get the same output as in C++)
    def myround(a):
        return int(np.floor(a + 0.5))
    # Computing the LPSD
    def computelpsd(d, f, r, m, L, navs, fs, win, winalpha, olap, Lmin, errtype, order, VERBOSE):
        nf    = len(f)       # Number of frequencies
        Ndata = len(d)       # Data length
        ENBW  = np.zeros(nf) # Initialize
        Sxx   = np.zeros(nf)
        Sstd  = np.zeros(nf)
        Smin  = np.zeros(nf)
        Smax  = np.zeros(nf)
        for jj in range(nf):
            # compute DFT exponent and window
            l    = np.floor(L[jj])
            w, _ = window(win, l, alpha = winalpha) # w: window
            # dft coefficients
            p = ( (2*np.pi*1j*m[jj]/l) ) * np.arange(l)
            C = w*np.exp(p)
            ovfact = 1. / (1. - olap / 100.) # Calculate the number of averages
            navg   = myround((((Ndata - l)) * ovfact) / l + 1)
            if VERBOSE: print('Computing frequency {}/{}:\t {} Hz.\t Navs:R: {}:{}. '.format(jj+1, nf, f[jj], navs[jj],  navs[jj]/navg))
            b     = np.zeros(navs[jj], dtype=complex)
            B     = 0.0
            if navg != 1:
                shift = (Ndata - l) / (navg - 1)
            else:
                shift = 1
            if (shift < 1):
                shift = 1
            start = 0
            for ii in range(navs[jj]): # Loop over the segments
                # start and end indices for the current segment
                istart = myround(start)
                start  = int(start + shift)
                iend   = int(istart+l)
                # get segment data
                seg = d[range(istart, iend)] 
                # detrend 
                dseg = detrend(seg, order)
                # make DFT
                a = np.sum(C*dseg)
                # Store it
                b[ii] = a*np.conjugate(a)
            # Do the Average
            B = 4.0*np.var(np.real(b))
            A = 2.0*np.mean(np.real(b))
            if navs[jj] == 1: # Scale the errors properly
                B = A**2
            else: 
                B = B/navs[jj]
            S1       = np.sum(w) 
            S12      = S1*S1
            S2       = np.sum(w*w)
            ENBW[jj] = fs*S2/S12
            Sxx[jj]  = A/fs/S2 # Scale to power
            Sstd[jj] = np.sqrt(B/S2**2/fs**2)
            # Choose between the errorbars type
            if errtype == 'std':
                Smin[jj], Smax[jj] = (Sstd[jj] for i in range(2))
            elif errtype == 'analytic':
                pnm      = norm.cdf([-1, 1])
                cL       = pnm[1]-pnm[0]
                Smin[jj] = navs[jj] * Sxx[jj] / gammaincinv(navs[jj]-1, (1+cL)/2)
                Smax[jj] = navs[jj] * Sxx[jj] / gammaincinv(navs[jj]-1, (1-cL)/2)
        return navs, Smax, Sxx, Smin, ENBW

    # Testing plot function
    def makePlots(f, data, lpsddata, Se, fs, navs, win, DOPLOT):
        if DOPLOT:
            import matplotlib.pyplot as plt
            import matplotlib as mpl
            fr, Sp, _ = psd(data, fs, navs=navs, win=win, olap=.50)
            lbls      = ['psd(data)', 'loglpsd(data)']
            errs      = [np.absolute(lpsddata - (lpsddata-Se[0])), np.absolute((lpsddata + Se[1])-lpsddata)]
            plt.figure()
            plt.loglog(fr, Sp, label=lbls[0])
            plt.errorbar(f, np.absolute(lpsddata), yerr=errs, fmt='.', label=lbls[1])
            plt.xlim(f[0], f[-1])
            plt.legend()
            plt.show()
        return None

    # Check if list
    if isinstance(d, list):
        NCH = len(d)
    elif isinstance(d, np.ndarray):
        NCH = 1
        d   = [d] # Put into a list
    else:
        exit('\t >>> Unknown data type. This function only takes list of numpy arrays, or numpy arrays. Please check again.')
        
    S, Se, ENBW, fvecs, Navgs = ([] for i in range(5)) # Init

    # Loop over the data 
    for kk in range(NCH):
        # Get inputs for LPSD
        f, r, m, L, K = ltf_plan(len(d[kk]), fs, olap, Lmin, Jdes, Kdes, flims)
        # Calculate the LPSD
        navs, Smax, Sxx, Smin, ENBWkk = computelpsd(d[kk], f, r, m, L, K, fs, win, winalpha, olap, Lmin, errrype, order, VERBOSE)
        # Append the result
        S.append(Sxx)
        Se.append([Smin, Smax])
        ENBW.append(ENBWkk)
        fvecs.append(f)
        Navgs.append(K)
        # Make test plots
        makePlots(f, d[kk], S[kk], Se[kk], fs, 25, win, DOPLOT)
    # prepare outputs
    if NCH == 1:
        return fvecs[0], S[0], Se[0], ENBW[0], Navgs[0]
    else:
        return fvecs, S, Se, ENBW, Navgs

# lpflogPSD function
def lpflogpsd(data, fs, dataY=[], win='blackman-harris', nperseg=-1, olap=50, nsigma=1, r = 6/7, fmin=-1, fmax=-1, onesided=False, DOPLOT=False, VERBOSE=True):
    """ logpsdbin (data, fs, dataY=[], win='blackman-harris', nperseg=-1, olap=50, nsigma=1, r = 6/7, fmin=-1, fmax=-1, onesided=False, DOPLOT=False, VERBOSE=True)

    A function to generate the binned power spectral density (or cross power PSD) given some time series data.

    For more details check https://link.aps.org/doi/10.1103/PhysRevLett.120.061101

    INPUTS: a) DATA:     The time series data.
            b) fs:       The sampling frequency of the data.
            c) DATAY:    The time series data, to perform the cross-spectra, if desired.
            d) win:      The window type. See the 'window' function of the same module.
            e) nperseg:  The number of data points per segment to perform the averages.
            f) olap:     The percentage of overplap between the segments,
            g) nsigma:   The number of sigmas of the errors.
            h) r:        The ratio.
            i) fmin:     Minumum frequency of the analysis.
            j) fmax:     Maximum frequency of the analysis.
            k) onesided: False-True flag to denote the input spectrum properties.
            l) DOPLOT:   False-True flag to produce a plot
            m) VERBOSE:  Verbose level.

    RETURN: freqs, navs, Smax, Sxx, Smin
            a) FREQS:    The frequencies.
            b) NAVS:     The number of averages per frequency bin.
            c) Smax:     The upper limits of the spectrum, according to the number of 'sigmas' specified.
            d) Sxx:      The actual spectrum produced.
            e) Smin:     The lower limits of the spectrum.

    NK 2019
    """
    p = norm.cdf([-nsigma, nsigma])
    c = p[1]-p[0]

    if nperseg == -1:
        Nseg = len(data)
    else:
        Nseg = nperseg

    if fmax != -1:
        fmax = fs/2.0

    if len(dataY) == 0:
        XSPEC = False
    else:
        XSPEC = True

    # Compute frequencies
    dT = 1/fs
    M  = 4

    f = M/(Nseg * dT)
    L     = Nseg
    kk    = 2
    freqs = f

    while f < fmax:
        N = r**(kk-2) * Nseg
        f = 2*M/(N * dT)
        L = np.append(L, N)
        kk    = kk + 1
        freqs = np.append(freqs, f)

    if fmin != -1 and fmin > freqs[0]:
        idx   = np.argwhere(fmin > 0.01)
        freqs = freqs[idx]
        L     = L[idx]

    if VERBOSE:
        for ii in range(freqs.shape[0]):
            print('f({}) = {}, N={}'.format(ii, freqs[ii], np.floor(L[ii])))

    # Process segments
    Ndata = len(data)
    order = 1
    nf    = len(freqs)
    ENBW  = np.zeros(nf)
    Sxx   = np.zeros(nf, dtype=complex)
    S     = np.zeros(nf, dtype=complex)
    navs  = np.zeros(nf)
    Smin  = np.zeros(nf, dtype=complex)
    Smax  = np.zeros(nf, dtype=complex)

    for jj in range(nf):
        if VERBOSE:
            print('computing frequency {} of {}: {} Hz'.format(jj, nf, freqs[jj]))

        # compute DFT exponent and window
        l     = np.floor(L[jj])
        wn, w = window(win, l, alpha = 0.2)

        # segment start indices
        sidx = np.arange(0, Ndata-l, l*olap/100)

        # dft coefficients
        p = ( (-2*np.pi*1j)/fs ) * np.arange(l)
        C = np.exp(freqs[jj]*p)
        navs[jj] = int(sidx.shape[0])
        A        = 0.0
        for ii in range(int(navs[jj])):
            # start and end indices for the current segment
            istart = int(sidx[ii])
            iend   = int(istart+l)
            # get segment data
            seg =  data[range(istart, iend)] # Detrend?

            # Check if cross-spectrum
            if XSPEC:
                segY = dataY[range(istart, iend)]
                segY = segY * w

            # window data
            seg = seg * w
            # make DFT
            a = np.sum(C*seg)

            if XSPEC:
                b = np.sum(C*segY)
                A = A + a*np.conjugate(b)
            else:
                A = A + a*np.conjugate(a)

        # scale and store results
        A2ns     = 2.0*A/navs[jj]
        S1       = np.sum(w)
        S12      = S1*S1
        S2       = np.sum(w*w)
        ENBW[jj] = fs*S2/S12

        if onesided:
            Sxx[jj]  = A2ns/fs/S2/2
        else:
            Sxx[jj]  = A2ns/fs/S2

        a        = navs[jj]-1
        z        = (1+c)/2
        Smin[jj] = navs[jj] * Sxx[jj] / gammaincinv(a, z)
        z        = (1-c)/2
        Smax[jj] = navs[jj] * Sxx[jj] / gammaincinv(a, z)

    if DOPLOT:
        import matplotlib.pyplot as plt
        import matplotlib as mpl
        # Do the psd with external code for refference
        if XSPEC:
            Sp, fr = cpsd(data, dataY, fs, nperseg)
            Sp     = np.absolute(Sp)
            lbls   = ['cpsd(data)', 'loglcpsd(data)']
        else:
            fr, Sp = welchpsd(data, fs, nperseg, win=win, olap=.50)
            lbls   = ['psd(data)', 'loglpsd(data)']

        errs  = [np.absolute(Sxx-Smin), np.absolute(Smax-Sxx)]
        plt.figure()
        plt.loglog(fr, Sp, label=lbls[0])
        plt.errorbar(freqs, np.absolute(Sxx), yerr=errs, fmt='o', label=lbls[1])
        plt.xlim(freqs[0], freqs[-1])
        plt.legend()
        plt.show()

    return freqs, navs, Smax, Sxx, Smin
