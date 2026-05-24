"""  One Euro Filter / Two Euro Filter

CHI 2012 Paper: https://gery.casiez.net/1euro/

The 1e filter (“one Euro filter”) is an adaptive first-order low-pass filter: it adapts the cutoff frequency of a
low-pass filter for each new sample according to an estimate of the signal’s speed, or more generally, its
derivative value. Even though noisy signals are often sampled at a fixed frequency, filtering can not always follow
the same pace, especially in event-driven systems.

A discrete time realization of a first order low-pass filter is given by Equation 1 where X_i and X̂_i denote the
raw and filtered data at time i and α is a smoothing factor in [0, 1]:
	X̂_i = α X_i + (1 − α) X̂_i−1		(1)

The first term of the equation is the contribution of new input data value, and the second term adds inertia from
previous values. As α decreases, jitter is reduced, but lag increases since the output responds more slowly to
changes in input. Since the contribution of older values exponentially decreases, a low-pass filter will have less
lag than a high n moving average filter.


The 1e filter is an adaptive first-order low-pass filter: it adapts the cutoff frequency of a low-pass filter for
each new sample according to an estimate of the signal’s speed, or more generally, its derivative value. Even
though noisy signals are often sampled at a fixed frequency, filtering can not always follow the same pace,
especially in event-driven systems. To accommodate possible fluctuations, we rewrite equation (1) to take into
account the actual time interval between samples.

Using a direct analogy with an electrical circuit, where a resistor in series with a capacitor defines a first
order low-pass filter, α can be computed as a function of the sampling period T_e and a time constant τ,
both expressed in seconds (Eq. 4). The resistor and capacitor values define the time constant (τ = RC) and the
corresponding cutoff frequency f_c , in Hertz, of the circuit (Equation 5).
	X^_i = α X_i + (1 − α) X̂_i−1     			(1)

	α = 1/(1 + τ/T_e)               			(4)

	τ = 1/(2 * pi * f_c)						(5)

	X̂_i =  (X_i + τ/T_e X̂_i−1) * 1/(1 + τ/T_e)  (6)

	f_c = f_cmin + β | dX̂_i /dt |				 (7)

The sampling period T_e (or its inverse, the sampling rate) can be automatically computed from timestamps, so
the cutoff frequency f_c is the only configurable parameter in equation (6).

As with any low-pass filter, decreasing f c reduces jitter, but increases lag. Finding a good trade-off between the
two is difficult since people are more sensitive to jitter at low speeds, and more sensitive to lag at high speeds.
This is why an adaptive cutoff frequency works well.

We found that a straight-forward linear relationship between cutoff frequency f_c and the absolute speed works well
(Equation 7). The speed (i.e the derivative dX̂_i/dt ) is computed from raw signal values using the sampling rate and then
low-pass filtered with a cutoff frequency chosen to avoid high derivative bursts caused by jitter.

There are three parameters:
	1) Sampling period T_e: default 1 second
    2) Intercept f_cmin in Equaiton 7
	3) Slope β shown in Equation 7
"""

import numpy as np
import math
import logging
from python_algo_production.signal.ema_filter import exp_moving_average

logger = logging.getLogger(__name__)	# create a logger for this module

### Utilit function for the Euro Filter
def cutoff_compute(f_cmin, beta, dvdt):
	f_c = f_cmin + beta * abs(dvdt)  		# f_c = f_cmin + β | dX̂_i /dt |	 (7)
	return f_c

def alpha_compute(rate, cutoff):
	tau = 1.0 / (2.0 * math.pi * cutoff)  	# τ = 1/(2 * pi * f_c)				(5)
	t_e = 1.0/ rate
	return 1.0 / (1.0 + tau / t_e)    		# α = 1/(1 + τ/T_e)               	(4)

### Basic 1D adaptive low-pass filter: OneEuro Filter
class OneEuroFilter:
    ## Initialization of the 1d euro filter
	def __init__(self,
		x0, 					# initial inputs: value
		f_cmin=1.0,  			# Intercept f_cmin in Equaiton 7
		beta=0.0, 				# Slope β shown in Equation 7
		rate=1.0):				# Sampling rate, default 1 hz

		# assigned parameters to the instance variables: use _var to indicate we want to keep them 'private'
		self.df_c = 1.0			# fixed cutoff frequency for derivative
		self.rate = rate  		# sampling period in seconds
		self.f_cmin = f_cmin  	# minimum cutoff frequency
		self.beta = beta   	 	# slope for speed

		# prepare for next step
		self.x_prev = x0
		self.dvdt_prev = 0.0  # previous derivative value

	## filter_step: pass a new item to the filter
	def filter_step(self, x):

		# compute the raw speed: needs filter to get rid of jitter
		dvdt_raw = (x - self.x_prev) * self.rate # unit: m/s
		alpha_dvdt = alpha_compute(self.rate, self.df_c)
		dvdt = exp_moving_average(alpha_dvdt, dvdt_raw, self.dvdt_prev)  # filter the speed to avoid jitter

		# f_c = f_cmin + β | dX̂_i /dt |	 (7)
		f_c = cutoff_compute(self.f_cmin, self.beta, dvdt)
		# 	α = 1/(1 + τ/T_e)               (4)
		alpha = alpha_compute(self.rate, f_c)

		# X̂_i = (X_i + τ/T_e X̂_i−1) * 1/(1 + τ/T_e)     (6)
		# filter and prepare for next step
		self.x_prev = exp_moving_average(alpha, x, self.x_prev)
		self.dvdt_prev = dvdt

		# return the filtered value
		return self.x_prev


	## filter: pass a batch of item to the filter
	def filter(self, x_array):
		if not isinstance(x_array, np.ndarray):
			raise TypeError("Input must be a numpy array.")

		# initialize the output array
		x_hat_array = np.zeros_like(x_array)
		# filter each sample
		for i, x in enumerate(x_array):
			x_hat_array[i] = self.filter_step(x)

		# return the filtered array
		return x_hat_array


### Extended 2D euro filter
# Expand x into a 2d vector v = (x, y)
# v^(t) = α v(t) + (1 − α) v^(t-1) and v(t) = x(t) + j*y(t) which yeilds
# x^(t) = α x(t) + (1 − α) x^(t-1) and
# y^(t) = α y(t) + (1 − α) y^(t-1)
# but dvdt = sqrt((dx/dt)^2 +  (dy/dt)^2)
class TwoEuroFilter(OneEuroFilter):
	def __init__(self,
			  	x0, y0,					# initial inputs: x and y value
				f_cmin=1.0,  			# Intercept f_cmin in Equaiton 7
				beta=0.0, 				# Slope β shown in Equation 7
				rate=1.0):  			# Sampling rate, default 1 hz

		super().__init__(x0, f_cmin, beta, rate)  # call the parent constructor
		# prepare for next step
		self.y_prev = y0


	def filter_step(self, xy):
		x = xy[0]
		y = xy[1]
		# compute the raw speed: needs filter to get rid of jitter
		dxdt = (x - self.x_prev) * self.rate
		dydt = (y - self.y_prev) * self.rate # unit: m/s
		dvdt_raw =  math.sqrt((dxdt)**2 + (dydt)**2)
		alpha_dvdt = alpha_compute(self.rate, self.df_c)
		dvdt = exp_moving_average(alpha_dvdt, dvdt_raw, self.dvdt_prev)  # filter the speed to avoid jitter

		# f_c = f_cmin + β | dX̂_i /dt |	 (7)
		f_c = cutoff_compute(self.f_cmin, self.beta, dxdt)
		# 	α = 1/(1 + τ/T_e)               (4)
		alpha = alpha_compute(self.rate, f_c)

		# X̂_i = (X_i + τ/T_e X̂_i−1) * 1/(1 + τ/T_e)     (6)
		# filter and prepare for next step
		self.x_prev = exp_moving_average(alpha, x, self.x_prev)
		self.y_prev = exp_moving_average(alpha, y, self.y_prev)
		self.dvdt_prev = dvdt

		# return the filtered value
		return self.x_prev, self.y_prev

	## filter: pass a batch of item to the filter
	def filter(self, xy_array):
		"""
		Apply the 2D EURO filter to batches of input values.

		Parameters:
			xy_array [n, 2]: row array of column input (x, y) values

		Returns:
			filtered row array of column input (x, y) values
		"""
		if not isinstance(xy_array, np.ndarray):
			raise TypeError("Input must be a numpy array.")

		# initialize the output array
		xy_hat_array = np.zeros_like(xy_array)

		for i in range(xy_array.shape[0]):
			xy = xy_array[i]
			xy_hat_array[i] = self.filter_step(xy)
		# return the filtered array
		return xy_hat_array