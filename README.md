# rampsclean
Code to remove baselines and glitches from RAMPS spectra.

RAMPS spectra have incompletely subtracted baselines and (at least in the first year) 
spikes from the VEGAS backend. Since the spectra are oversampled we median filter to 
eliminate spikes and use the local-standard-deviation (i.e. the standard deviation 
calculated in rolling windows) to identify regions of signal, mask those regions for
fitting a smooth baseline, and return the final cleaned spectrum.

Here is a synthetic spectrum simulating an NH3 (1-1) line profile as seen by RAMPS.

<img src="example_graphics/total_synthetic.png?raw=True" alt="Synthetic" width ="300"/>

After median filtering and downsampling to remove the spikes, we identify portions of 
the spectrum where the local-standard-deviation is much higher than typical. This approach
works much better than identifying signal based on the intensity/strength of the signal in 
the case where the amplitude of the baseline is large (at least in parts of the spectrum).

<img src="example_graphics/local-stddev.png?raw=True" alt="Synthetic" width ="300"/>

We mask out the regions where we have identified signal and fit a smooth baseline to the r
est of the spectrum. In this case we are using a spline fit.

<img src="example_graphics/baseline-fit.png?raw=True" alt="Synthetic" width ="300"/>

We can use the same local-standard-deviation to identify regions of real signal in the cleaned
spectrum. The procedure is tested by comparing the integrated intensity of this signal against
the true integrated intensity of the injected/synthetic signal.

<img src="example_graphics/moment-mask.png?raw=True" alt="Synthetic" width ="300"/>




