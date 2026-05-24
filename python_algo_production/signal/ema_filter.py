"""
This is a Python module that implements exponential moving average (ema) filters.
The weights exponentially decay for past samples, hence the name "exponential".
It is esentially a a First-order Infinite Impulse Response (IIR) Low-Pass Filter.

The fixed α controls the smoothness:
* α close to 1 means less smoothing (more responsive).
* α close to 0 means more smoothing (more lag).

Basic formula of exponential moving average (EMA) filter:

	X^_i = α X_i + (1 − α) X̂_i−1
"""

import numpy as np
import math
import logging
logger = logging.getLogger(__name__)  # create a logger for this module

# basic formula of exponential moving average filter:
def exp_moving_average(alpha, x, x_prev):
    return alpha * x + (1 - alpha) * x_prev

### Basic 1D exponential moving average filter
class OneEmaFilter:
	'''
	Exponential Moving Average Filter (EMA) for a single dimension.

	Parameters:
		alpha: smoothing factor (0 < alpha <= 1)
		x0: initial value
	'''
	def __init__(self, x0, alpha=0.1):
		if not (0 < alpha <= 1):
			raise ValueError("Alpha must be in the range (0, 1].")
		self.alpha = alpha

		# prepare for next step
		self.x_prev = x0

	def filter_step(self, x):
		"""
		Apply the EMA filter to a new input value.

		Parameters:
			x: new input value

		Returns:
			filtered value
		"""
		self.x_prev = exp_moving_average(self.alpha, x, self.x_prev)
		return self.x_prev

	def filter(self, x_array):
		"""
		Apply the EMA filter to a batch of input values.

		Parameters:
			x_array: array of new input values

		Returns:
			array of filtered values
		"""
		if not isinstance(x_array, np.ndarray):
			raise TypeError("Input must be a numpy array.")

		x_hat_array = np.zeros_like(x_array)
		for i, x in enumerate(x_array):
			x_hat_array[i] = self.filter_step(x)

		return x_hat_array

### Extended 2D exponential moving average filter
# Expand x into a 2d vector v = (x, y)
# v^(t) = α v(t) + (1 − α) v^(t-1) and v(t) = x(t) + j*y(t) which yeilds
# x^(t) = α x(t) + (1 − α) x^(t-1) and
# y^(t) = α y(t) + (1 − α) y^(t-1)
class TwoEmaFilter (OneEmaFilter):
	def __init__(self, x0, y0, alpha=0.1):
		super().__init__(x0, alpha)
		# prepare for next step
		self.y_prev = y0


	def filter_step(self, xy):
		"""
		Apply the 2D EMA filter to new input values.
		Parameters:
			xy: new input (x,y) value
		Returns:
			filtered x and y
		"""
		x = xy[0]
		y = xy[1]
		self.x_prev = exp_moving_average(self.alpha, x, self.x_prev)
		self.y_prev = exp_moving_average(self.alpha, y, self.y_prev)
		# return the filtered item
		return self.x_prev, self.y_prev

	def filter(self, xy_array):
		"""
		Apply the 2D EMA filter to batches of input values.

		Parameters:
			xy_array [n, 2]: row array of column input (x, y) values

		Returns:
			filtered row array of column input (x, y) values
		"""
		if not (isinstance(xy_array, np.ndarray)):
			raise TypeError("Input xy_array must be numpy array.")

		xy_hat_array = np.zeros_like(xy_array)
		for i in range(xy_array.shape[0]):
			xy = xy_array[i]
			xy_hat_array[i] = self.filter_step(xy)
		# return the filtered array
		return xy_hat_array