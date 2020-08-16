"""These are the tools from disentanglementlib, 
extracted to what is needed (disentanglementlib uses 
tf1 which might cause problems)

https://github.com/google-research/disentanglement_lib/tree/master/disentanglement_lib/methods
"""
import tensorflow as tf
import math

def gaussian_log_density(samples, mean, log_var):
	pi = tf.constant(math.pi)
	normalization = tf.math.log(2. * pi)
	inv_sigma = tf.math.exp(-log_var)
	tmp = (samples - mean)
	return -0.5 * (tmp * tmp * inv_sigma + log_var + normalization)


def total_correlation(z, z_mean, z_logvar):
	"""Estimate of total correlation on a batch.
	We need to compute the expectation over a batch of: E_j [log(q(z(x_j))) -
	log(prod_l q(z(x_j)_l))]. We ignore the constants as they do not matter
	for the minimization. The constant should be equal to (num_latents - 1) *
	log(batch_size * dataset_size)
	Args:
		z: [batch_size, num_latents]-tensor with sampled representation.
		z_mean: [batch_size, num_latents]-tensor with mean of the encoder.
		z_logvar: [batch_size, num_latents]-tensor with log variance of the encoder.
	Returns:
		Total correlation estimated on a batch.
	"""
	# Compute log(q(z(x_j)|x_i)) for every sample in the batch, which is a
	# tensor of size [batch_size, batch_size, num_latents]. In the following
	# comments, [batch_size, batch_size, num_latents] are indexed by [j, i, l].
	log_qz_prob = gaussian_log_density(
			tf.expand_dims(z, 1), tf.expand_dims(z_mean, 0),
			tf.expand_dims(z_logvar, 0))
	# Compute log prod_l p(z(x_j)_l) = sum_l(log(sum_i(q(z(z_j)_l|x_i)))
	# + constant) for each sample in the batch, which is a vector of size
	# [batch_size,].
	log_qz_product = tf.reduce_sum(
			tf.reduce_logsumexp(log_qz_prob, axis=1, keepdims=False),
			axis=1,
			keepdims=False)
	# Compute log(q(z(x_j))) as log(sum_i(q(z(x_j)|x_i))) + constant =
	# log(sum_i(prod_l q(z(x_j)_l|x_i))) + constant.
	log_qz = tf.reduce_logsumexp(
			tf.reduce_sum(log_qz_prob, axis=2, keepdims=False),
			axis=1,
			keepdims=False)

	return log_qz - log_qz_product # MODIFIED for sum reduction reduction
