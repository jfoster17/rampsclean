# rampsclean
Code to remove baselines and glitches from RAMPS spectra.

RAMPS spectra have incompletely subtracted baselines and (at least in the first year) 
spikes from the VEGAS backend. Since the spectra are oversampled we median filter to 
eliminate spikes and use the local-standard-deviation (i.e. the standard deviation 
calculated in rolling windows) to identify regions of signal, mask those regions for
fitting a smooth baseline, and return the final cleaned spectrum.

Here is a synthetic spectrum simulating an NH3 (1-1) line profile as seen by RAMPS.

<img src="example_graphics/total_synthetic.png?raw=True" alt="Synthetic" width ="300"/>

After median filtering and downsampling to remove the spikes, we mask out the regions where 
there is signal by identifying where the local-standard-deviation is much higher than typical
and fit a smooth baseline to the rest of the spectrum. In this case we are using a spline fit.

<img src="example_graphics/baseline-fit.png?raw=True" alt="Synthetic" width ="300"/>

We can use the same local-standard-deviation to identify regions of real signal in the cleaned
spectrum. The procedure is tested by comparing the integrated intensity of this signal against
the true integrated intensity of the injected/synthetic signal.

<img src="example_graphics/moment-mask.png?raw=True" alt="Synthetic" width ="300"/>




