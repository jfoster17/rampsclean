import numpy as np
from numpy.polynomial import Polynomial as P 
from astropy.modeling import models 
import moments
import matplotlib.pyplot as plt
import os,sys

class SyntheticSpectrum:
    """
    Class to make a synthetic spectrum.
    
    Make a synthetic NH3 spectrum with properties 
    similar to what is observed with RAMPS. This 
    object will contain the synthetic spectrum, the 
    parameters used to make said spectrum and a 
    measurement of the noise-free integrated intensity 
    of the spectrum (for testing purposes).
    """
    
    def __init__(self,parameters=None,**kwargs):
        if not parameters:
            self.p = {
                "spec_length" : 16384, #Spectrum properties
                "noise_level" : 0.02, #Noise properties
                "baseline_poly_order" : 2,  #Baseline properties
                "baseline_poly_params" : np.array([-0.1,+1e-6,-5e-10,+1e-13]),
                "do_random_baseline" : False,
                "nh3_amplitude" : 4.0, #NH3 spectrum properties
                "nh3_width" : 40.,
                "nh3_position" : 2000.,
                "nh3_offset" : 300.,
                "num_spikes" : 10.,
                "spikes_amp"  : 4.,
            }
        else:
            self.p = parameters
        self.noisy_spectrum = self.make_noisy_spectrum(**kwargs)
        self.baseline_spectrum,self.baseline_poly = self.make_baseline(self.p['do_random_baseline'])
        self.nh3_spectrum = self.make_nh3_spectrum()
        self.spikes_spectrum,self.spike_channels,self.spike_amps = self.make_spikes()
        
    def calculate_integrated_intensity(self):
        """
        Calculate the integrated intensity for the test spectrum
        """
        self.noise_free = self.nh3_spectrum
        mom0 = moments.get_integrated_intensity(self.noise_free,0) #no noise
        self.mom0 = mom0
        return(mom0)
        
    def generate_spectrum(self,do_noise=True,do_base=True,do_nh3=True,do_spikes=True,**kwargs):
        """
        Sum together components of spectrum
        
        Select which components to sum together
        and return total spectrum for analysis
        """
        total = np.zeros(self.p['spec_length'])
        if do_noise:
            total += self.noisy_spectrum
        if do_base:
            total += self.baseline_spectrum
        if do_nh3:
            total += self.nh3_spectrum
        if do_spikes:
            total += self.spikes_spectrum
        self.total_spectrum = total
        if "outdir" in kwargs:
            try:
                os.mkdir(kwargs["outdir"])
            except OSError:
                pass
            plt.figure()
            plt.plot(total,color="blue",alpha=0.5,)
            plt.xlim(0,len(total))
            plt.title("Total Synthetic")
            plt.savefig(kwargs["outdir"]+"/total_synthetic.png")
        return(self.total_spectrum)
        
    def make_noisy_spectrum(self,**kwargs):
        """
        Generate noise for a fake spectrum.
        
        Makes a fake spectrum of length spec_length with 
        specified Gaussian noise ~N(0,k**2)
        """
        k = self.p['noise_level']
        noisy_spectrum = k * np.random.randn(self.p['spec_length'])
        return(noisy_spectrum)
        
    def make_baseline(self,do_random_baseline=False):
        """
        Make a polynomial baseline of order n.
        
        Make a polynomial baseline either with specified 
        coefficient array or with a randomly generated 
        (but sensible) set of coefficients.
        Return the polynomial (encoding the parameters)
        and the baseline (polynomial evaluated over spectrum)
        """
        b = self.p['baseline_poly_params']
        n = self.p['baseline_poly_order']
        if do_random_baseline:
            b = np.random.rand(n+1)
            #If we just have random coefficients in b
            #then higher-order terms will hugely dominate
            for i,bb in enumerate(b):
                bb = bb/(self.p['spec_length']**i)
        #print(b)
        poly = P(b)
        baseline = poly(np.arange(self.p['spec_length']))
        return(baseline,poly)
        
    def make_nh3_spectrum(self):
        """
        Make a snythetic NH3 line with specified parameters.

        These are groups of five Gaussians with the central 
        peaks having amplitude a (and satellites have 
        amplitude a/3), widths of w and an arbitrary 
        central position p (with satellites at fixed 
        p-2q, p-q, p+q, p+2q)
        """
        a = self.p['nh3_amplitude']
        w = self.p['nh3_width']
        p = self.p['nh3_position']
        q = self.p['nh3_offset']
        gcen = models.Gaussian1D(amplitude=a,    mean=p, stddev=w)
        g_s1 = models.Gaussian1D(amplitude=a/3., mean=p-q, stddev=w)
        g_s2 = models.Gaussian1D(amplitude=a/3., mean=p-2*q, stddev=w)
        g_s3 = models.Gaussian1D(amplitude=a/3., mean=p+q, stddev=w)
        g_s4 = models.Gaussian1D(amplitude=a/3., mean=p+2*q, stddev=w)
        emp = np.arange(self.p['spec_length'])
        nh3_spectrum = gcen(emp)+g_s1(emp)+g_s2(emp)+g_s3(emp)+g_s4(emp)
        return(nh3_spectrum)
        
    def make_spikes(self):
        """
        Make some spikes/glitches similar to what is seen in RAMPS

        Positions are random (c) and amplitudes (z) are exponential 
        """
        amplitude = self.p['spikes_amp']
        num = self.p['num_spikes']
        c = np.random.randint(0,self.p['spec_length'],num)
        z = np.random.exponential(scale=amplitude,size=num)
        #print(c)
        #print(z)
        emp = np.zeros(self.p['spec_length'])
        for ci,zi in zip(c,z):
            emp[ci] += zi
        return(emp,c,z)