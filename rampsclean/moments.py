"""
Do basic moment calculations on spectrum.

Currently:
Moment 0 (integrated intensity) and error

"""
import numpy as np
import numpy.ma as ma

def get_integrated_intensity(input_spectrum,noise_estimate):
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
    """
    mom0 = ma.sum(input_spectrum)
    num_channels = ma.count(input_spectrum)
    mom0_err = np.sqrt(num_channels)*noise_estimate
    return(mom0,mom0_err)