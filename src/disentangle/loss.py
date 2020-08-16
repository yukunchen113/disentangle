import tensorflow as tf
import numpy as np

@tf.function
def cross_entropy(inputs, pred, epsilon=1e-7):
	"""
	Applys cross entropy per pixel between the inputs and the predictions

	we need to flatten our data, so we can reduce it per batch.

	Example:
		>>> a = np.random.normal(3,4,5)
		>>> b = np.random.normal(3,4,5)
		>>> res1 = cross_entropy(a,b).numpy()
	"""
	inputs = tf.keras.layers.Flatten()(inputs)
	pred = tf.keras.layers.Flatten()(pred)

	pred = tf.clip_by_value(pred, epsilon, 1-epsilon)
	loss = -tf.reduce_sum(
			inputs * tf.math.log(pred) + 
			(1-inputs) * tf.math.log(1-pred), 
			axis=1)
	return loss


@tf.function
def kl_divergence_with_normal(mean, log_var):
	"""
	gets kl divergence for a given men and variance, with the normal distribution
	log_var is the vector representing the diagonal of the covariance
	matrix. The covariances are assumed to be zero.

	Example:
		>>> a = np.random.normal(3,4,5)
		>>> b = np.random.normal(3,4,5)
		>>> res1 = kl_divergence_with_normal(a,b).numpy()
	"""
	var = tf.math.exp(log_var)
	return 0.5*(var+tf.math.square(mean)-1-log_var)


@tf.function
def kl_divergence_between_gaussians(mean1, logvar1, mean2, logvar2):
	"""Calculates KL Divergence between two gaussian distributions, given the
	mean, and log variance
	
	Args:
		mean1 (tensorflow tensor): The mean of distribution 1
		logvar1 (tensorflow tensor): The log variance of distribution 1
		mean2 (tensorflow tensor): The mean of distribution 2
		logvar2 (tensorflow tensor): The log variance of distribution 2
	"""
	var1 = tf.math.exp(logvar1)
	var2 = tf.math.exp(logvar2)
	kl_loss = 0.5*(logvar2 - logvar1 - 1 + (var1+tf.math.square(mean1-mean2))/var2)
	return kl_loss