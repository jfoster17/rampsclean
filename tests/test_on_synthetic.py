import rampsclean.synthetic_spectrum as snythetic_spectrum
import rampsclean.moments as moments
import rampsclean.clean_spectrum as clean_spectrum
import numpy.ma as ma

def test_mom0():
    a = snythetic_spectrum.SyntheticSpectrum()
    test_spectrum = a.generate_spectrum()
    noise_free_mom0,noise_free_mom0_err = a.calculate_integrated_intensity()
    print(noise_free_mom0)
    cleaned_spectrum = clean_spectrum.baseline_and_deglitch(test_spectrum)
    signal_spectrum,noise_estimate = moments.identify_signal_estimate_noise(cleaned_spectrum)
    print(ma.count(signal_spectrum))
    cleaned_mom0,cleaned_mom0_err = moments.get_integrated_intensity(signal_spectrum,noise_estimate,downsample_fact=7)
    print(cleaned_mom0)
    print(cleaned_mom0_err)
    assert abs(cleaned_mom0-noise_free_mom0) < 3*cleaned_mom0_err
    