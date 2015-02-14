import rampsclean.synthetic_spectrum as snythetic_spectrum
import rampsclean.moments as moments
import rampsclean.clean_spectrum as clean_spectrum
import numpy.ma as ma
import numpy as np

def check_mom0(a,**kwargs):
    """
    Run full process on a test spectrum and compare integrated intensity
    """
    test_spectrum = a.generate_spectrum(**kwargs)
    noise_free_mom0,noise_free_mom0_err = a.calculate_integrated_intensity()
    #print(noise_free_mom0)
    cleaned_spectrum = clean_spectrum.baseline_and_deglitch(test_spectrum,filter_width=7,basetype="spline",**kwargs)
    signal_spectrum,noise_estimate = moments.identify_signal_estimate_noise(cleaned_spectrum,**kwargs)
    cleaned_mom0,cleaned_mom0_err = moments.get_integrated_intensity(
                                    signal_spectrum,noise_estimate,downsample_fact=7)
    assert abs(cleaned_mom0-noise_free_mom0) < 5*cleaned_mom0_err
    

def test_mom0_highSNR():
    parameters = {
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
    a = snythetic_spectrum.SyntheticSpectrum(parameters=parameters,outdir="high_snr")
    check_mom0(a,outdir="high_snr")
    
def test_mom0_medSNR():
    parameters = {
        "spec_length" : 16384, #Spectrum properties
        "noise_level" : 0.20, #Noise properties
        "baseline_poly_order" : 2,  #Baseline properties
        "baseline_poly_params" : np.array([-0.1,+1e-6,-5e-10,+1e-13]),
        "do_random_baseline" : False,
        "nh3_amplitude" : 3.0, #NH3 spectrum properties
        "nh3_width" : 50.,
        "nh3_position" : 4000.,
        "nh3_offset" : 300.,
        "num_spikes" : 10.,
        "spikes_amp"  : 4.,
    }
    
    a = snythetic_spectrum.SyntheticSpectrum(parameters=parameters,outdir="med_snr")
    check_mom0(a,outdir="med_snr")
    
    
def test_mom0_lowSNR():
    parameters = {
        "spec_length" : 16384, #Spectrum properties
        "noise_level" : 0.50, #Noise properties
        "baseline_poly_order" : 2,  #Baseline properties
        "baseline_poly_params" : np.array([-0.1,+1e-6,-5e-10,+1e-13]),
        "do_random_baseline" : False,
        "nh3_amplitude" : 3.0, #NH3 spectrum properties
        "nh3_width" : 50.,
        "nh3_position" : 4000.,
        "nh3_offset" : 300.,
        "num_spikes" : 10.,
        "spikes_amp"  : 4.,
    }
    
    a = snythetic_spectrum.SyntheticSpectrum(parameters=parameters,outdir="low_snr")
    check_mom0(a,outdir="low_snr")
    
def test_mom0_verylowSNR():
    parameters = {
        "spec_length" : 16384, #Spectrum properties
        "noise_level" : 0.50, #Noise properties
        "baseline_poly_order" : 2,  #Baseline properties
        "baseline_poly_params" : np.array([-0.1,+1e-6,-5e-10,+1e-13]),
        "do_random_baseline" : False,
        "nh3_amplitude" : 2.0, #NH3 spectrum properties
        "nh3_width" : 50.,
        "nh3_position" : 4000.,
        "nh3_offset" : 300.,
        "num_spikes" : 10.,
        "spikes_amp"  : 4.,
    }

    a = snythetic_spectrum.SyntheticSpectrum(parameters=parameters,outdir="verylow_snr")
    check_mom0(a,outdir="verylow_snr")
    
    
def test_mom0_broadlines():
    parameters = {
        "spec_length" : 16384, #Spectrum properties
        "noise_level" : 0.2, #Noise properties
        "baseline_poly_order" : 2,  #Baseline properties
        "baseline_poly_params" : np.array([-0.1,+1e-6,-5e-10,+1e-13]),
        "do_random_baseline" : False,
        "nh3_amplitude" : 3.0, #NH3 spectrum properties
        "nh3_width" : 100.,
        "nh3_position" : 2000.,
        "nh3_offset" : 300.,
        "num_spikes" : 10.,
        "spikes_amp"  : 4.,
    }
    a = snythetic_spectrum.SyntheticSpectrum(parameters=parameters,outdir="broad_line")
    check_mom0(a,outdir="broad_line")
