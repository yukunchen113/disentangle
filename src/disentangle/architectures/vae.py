import tensorflow as tf 
from disentangle.loss import kl_divergence_with_normal
from disentangle.other_library_tools.disentanglementlib_tools import gaussian_log_density, total_correlation 
from . import encoder as enc
from . import decoder as dec
class VariationalAutoencoder(tf.keras.Model):
	def __init__(self, name="variational_autoencoder", is_create_default_vae=True, **kwargs):
		# default model
		super().__init__(name=name)
		if is_create_default_vae:
			self.create_default_vae(**kwargs)

	def create_default_vae(self, **kwargs):
		self.create_encoder_decoder_64(**kwargs)

	def _setup(self):
		# dynamically changing
		self.latest_sample = None
		self.latest_mean = None
		self.latest_logvar = None

	def call(self, inputs):
		self.latest_sample, self.latest_mean, self.latest_logvar = self.encoder(inputs)
		reconstruction = self.decoder(self.latest_sample)
		self.add_loss(self.regularizer(self.latest_sample, self.latest_mean, self.latest_logvar))
		return reconstruction

	def get_latent_space(self):
		return self.latest_sample, self.latest_mean, self.latest_logvar

	def regularizer(self, sample, mean, logvar):
		return kl_divergence_with_normal(mean, logvar)
	
	def set_encoder_decoder(self, encoder, decoder):
		self._encoder = encoder
		self._decoder = decoder

	def create_encoder_decoder_64(self, **kwargs):
		# default encoder decoder pair:
		self._encoder = enc.GaussianEncoder64(**kwargs)
		self._decoder = dec.Decoder64(**kwargs)
		self._setup()

	def create_encoder_decoder_256(self, **kwargs):
		# default encoder decoder pair:
		self._encoder = enc.GaussianEncoder256(**kwargs)
		self._decoder = dec.Decoder256(**kwargs)
		self._setup()

	def create_encoder_decoder_512(self, **kwargs):
		# default encoder decoder pair:
		self._encoder = enc.GaussianEncoder512(**kwargs)
		self._decoder = dec.Decoder512(**kwargs)
		self._setup()
	
	def get_config(self):
		conf_params = {
			"encoder":self.encoder.get_config(),
			"decoder":self.decoder.get_config()}
		return conf_params

	@property
	def num_latents(self):
		return self.encoder.num_latents

	@property
	def shape_input(self):
		return self.encoder.shape_input

	@property
	def encoder(self):
		return self._encoder

	@property
	def decoder(self):
		return self._decoder

class BetaVAE(VariationalAutoencoder):
	def __init__(self, beta, name="BetaVAE", **kwargs):
		super().__init__(name=name, **kwargs)
		self.beta = beta

	def regularizer(self, sample, mean, logvar):
		# regularization uses disentanglementlib method
		kl_loss = self.beta*kl_divergence_with_normal(mean, logvar)
		kl_loss = tf.reduce_sum(kl_loss,1)
		return kl_loss
	
	def get_config(self):
		config_param = {
			**super().get_config(),
			"beta":self.beta}
		return config_param

class BetaTCVAE(BetaVAE):
	def __init__(self, beta, name="BetaTCVAE", **kwargs):
		super().__init__(beta=beta, name=name, **kwargs)

	def regularizer(self, sample, mean, logvar):
		# regularization uses disentanglementlib method
		kl_loss = kl_divergence_with_normal(mean, logvar)
		kl_loss = tf.reduce_sum(kl_loss,1)
		tc = (self.beta - 1) * total_correlation(sample, mean, logvar)
		return tc + kl_loss

class JointVAE(BetaVAE):
	def __init__(self, beta, name="JointVAE", **kwargs):
		super().__init__(name=name, **kwargs)


	def regularizer(self, sample, mean, logvar):
		# regularization uses disentanglementlib method
		pass
	
	def get_config(self):
		config_param = {
			**super().get_config(),

			}
		return config_param