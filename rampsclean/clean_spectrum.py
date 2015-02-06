"""
Remove a smooth baseline and de-glitch a specrum.

This code removes a smooth baseline and (optionally)
spikes/glitches from a spectrum. It is optimized for
RAMPS data from VEGAS (2013-2015), when the spectrum
is highly oversampled. As such, the spike removal is
done by median down-sampling the spectrum, and by 
default the baseline fitting is also done on a 
down-sampled spectrum to speed up the process and 
prevent memory problems when running in parallel.

The key to this analysis is the use of the 
local-standard-deviation array (y). This array 
calculates the sample standard deviation for each 
point, yi, as the sample standard deviation within
[yi-ww, yi+ww] for some window width ww. We expect
that 
y ~ N(k,sy**2)
where k is the per-channel noise and 
sy ~ k/(sqrt(2 ww))
from the asymptotic standard error on calculating
the sample standard deviation from a sample of size
2 ww.

We can therefore mask the spectrum based on where
y is greater than k + 3 sy (i.e. more than 3-sigma
deviant). This works as long as the window width
is large compared to the width of real features
and small compared to variations in the baseline. 
"""
from scipy.interpolate import UnivariateSpline 
from numpy.polynomial import Polynomial as P
import numpy as np
import scipy.ndimage as im
import numpy.ma as ma
import os,sys
import matplotlib.pyplot as plt


def baseline_and_deglitch(spec,filter_width=7,ww=20,**kwargs):
    """
    Do baseline subtraction and remove spikes via median filter
    """
    #Median filter both removes spikes and increases speed
    downsampled_spec = im.median_filter(spec,filter_width)[::filter_width]
    y = make_local_stddev(downsampled_spec,ww=ww)
    k_est = np.median(y)
    no_signal_spec = mask_spectrum(y,ww,downsampled_spec,keep_signal=False,**kwargs)
    if basetype=="spline":
        baseline = get_spline_baseline(no_signal_spec)
    elif basetype=="poly":
        baseline = get_poly_baseline(no_signal_spec,k_est)
    if "outdir" in kwargs:
        try:
            os.mkdir(kwargs["outdir"])
        except OSError:
            pass
        plt.figure()
        plt.plot(downsampled_spec,color="blue",alpha=0.1,)
        plt.plot(no_signal_spec,color="blue",alpha=0.6,)
        plt.plot(baseline,color="red",alpha=1.0,)
        plt.xlim(0,len(downsampled_spec))
        plt.ylabel("Spectral Intensity")
        plt.xlabel("Spectral Pixel")
        plt.title("Baseline Fit")
        plt.savefig(kwargs["outdir"]+"/baseline-fit.png")
    final_spec = downsampled_spec - baseline
    return(final_spec)
    
def get_spline_baseline(mspec):
    """
    Spline fit a baseline on the masked spectrum
    """
    xxx = np.arange(mspec.size)
    w = ma.getmaskarray(mspec)
    spl = UnivariateSpline(xxx, mspec, w=~w)
    fit_baseline = spl(xxx)
    return(fit_baseline)
    
def get_polynomial_baseline(mspec,k_est,debug=True):
    """
    Fit for the best polynomial baseline according to BIC
    
    Search polynomial fits from order 0 to order 7 and
    use the BIC to select the best fit. This fit should 
    also have an rms error within a factor of two of 
    the estimated noise (k_est). If this is not the case
    then the baseline fit is most likely bad. 
    """
    d = np.arange(0,7)
    rms_err = np.zeros(d.shape)
    
    xx = np.arange(mspec.size)
    yy = mspec
    
    for i in range(len(d)):
        p = ma.polyfit(xx,yy,d[i])
        basepoly = P(p[::-1])
        rms_err[i] = np.sqrt(np.sum((basepoly(xx) - yy) **2) / len(yy))
    BIC = np.sqrt(len(yy)) * rms_err / k_est + 1 * d * np.log(len(yy))
    if debug:
        fig = plt.figure(figsize=(12,5))
        ax = fig.add_subplot(211)
        ax.plot(d,rms_err,'-k',label='rms-err')
        ax.legend(loc=2)
        ax.axhline(k_est)
        ax = fig.add_subplot(212)
        ax.plot(d,BIC,'-k',label='BIC')
        ax.legend(loc=2)
        plt.savefig("debugplot.pdf")

        
        
    
    

def mask_spectrum(y,ww,spec,stddevlev=3,keep_signal=True,**kwargs):
    """
    Mask spectrum based on y-array
    
    Identify regions with significant signal based on y-array
    and standard deviation level. Mask significant signal either 
    in or out, depending on flag.
    """
    k_est = np.median(y)
    std_y = k_est/(np.sqrt(2*ww*2)) #extra 2 here because ww is half the real window 
    upperlim = k_est+std_y*stddevlev
    if keep_signal:
        mspec = ma.masked_where(y < upperlim,spec)
    else:
        mspec = ma.masked_where(y > upperlim,spec)
    if "outdir" in kwargs:
        try:
            os.mkdir(kwargs["outdir"])
        except OSError:
            pass
        plt.figure()
        plt.plot(y,color="blue",alpha=0.5,)
        plt.xlim(0,len(y))
        plt.axhline(upperlim,color='red',ls=":")
        plt.ylabel("Local Standard Deviation (y-array)")
        plt.xlabel("Spectral Pixel")
        plt.title("Local Standard Deviation Masking")
        plt.savefig(kwargs["outdir"]+"/local-stddev.png")
        
    return(mspec)

def rolling_window(a,window):
    """
    Magic code to quickly create a second dimension
    with the elements in a rolling window. This
    allows us to apply numpy operations over this
    extra dimension MUCH faster than using the naive approach.
    """
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)    
    strides = a.strides+(a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

def make_local_stddev(orig_spec,ww=300):
    """
    Make an array that encodes the local standard
    deviation within a window of width ww
    """
    import my_pad
    ya = rolling_window(orig_spec,ww*2)
    y = my_pad.pad(np.std(ya,-1),(ww-1,ww),mode='edge')
    return(y)