import rampsclean.synthetic_spectrum as snythetic_spectrum
import rampsclean.moments
import rampsclean.clean_spectrum

def test_mom0():
    a = snythetic_spectrum.SyntheticSpectrum()
    test_spectrum = a.generate_spectrum()
    noise_free_mom0 = a.calculate_integrated_intensity()
    print(noise_free_mom0)
    cleaned_spectrum = clean_spectrum.baseline_and_deglitch(test_spectrum)
    
    assert 3*5 == 15
    