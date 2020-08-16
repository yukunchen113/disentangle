"""
This module contains general tools to process and manipulate arrays.

"""
import numpy as np

def shuffle(*args, **kwargs):
	"""
	Takes in arrays of the same length in the 0th axis and shuffles them the same way

	Args:
		*args: numpy arrays.
		**kwargs: numpy arrays.

	Returns:
		arrays in the same order as been put in.
	"""
	args = list(args) + list(kwargs.values())
	idx = np.arange(args[0].shape[0])
	np.random.shuffle(idx)
	new_data = []
	for i in args:
		new_data.append(i[idx])
	return new_data
