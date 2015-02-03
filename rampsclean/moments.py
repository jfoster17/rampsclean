"""
Do basic moment calculations on spectrum.

Currently:
Moment 0 (integrated intensity) and error

"""
import numpy as np
import numpy.ma as ma
import clean_spectrum

def identify_signal_estimate_noise(input_spectrum,ww=20):
    """
    Use the local-standard-deviation to mask signal
    
    """
    y = clean_spectrum.make_local_stddev(input_spectrum,ww=ww)
    k_est = np.median(y)
    signal_spec = clean_spectrum.mask_spectrum(y,ww,input_spectrum,keep_signal=True)
    return(signal_spec,k_est)


def get_integrated_intensity(input_spectrum,noise_estimate,downsample_fact=1.):
    """
    Calculate the integrated intensity.
    
    Calculate the integrated intensity of an input
    spectrum of the full range of that spectrum. Input
    should be masked to perform the calculation over
    the relevant portion only. An estimate for the error
    is returned as well. This is most accurate if called
    with a pre-calculated noise_estimate, as one generally
    cannot calculate the noise in the spectrum from the
    signal-only portion of the spectrum.
    
    downsample_fact = amount by which the spectrum has been
                      downsampled (since mom0 here is in
                      units of channels).
    """
    mom0 = ma.sum(input_spectrum)
    num_channels = ma.count(input_spectrum)
    mom0_err = np.sqrt(num_channels)*noise_estimate
    return(mom0*downsample_fact,mom0_err*downsample_fact)