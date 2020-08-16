import copy
import tensorflow as tf
from . import architecture_params as ap
from . import base
from . import network

class Decoder(network.DeconvolutionalNeuralNetwork):
	def __init__(self, layer_param, num_latents, shape_image, activation=None, **kwargs):
		"""Base class for decoders
		
		Args:
			layer_param (list): layer size specs
			shape_image (list): the shape of the input image (not including batch size)
			activation (dict): this is a dictionary of activation
		    shape_before_flatten (None, list): shape of activation before flattening.
		"""
		# default activations
		if activation == None:
			activation = ap.default_decoder_activations

		# add additional attributes
		self.shape_image = shape_image
		self.num_latents = num_latents

		super().__init__(*layer_param, activation=activation, shape_input=[num_latents], **kwargs)
		
		self._config_param = dict(
					layer_param=layer_param, 
					shape_input=num_latents,
					shape_image=shape_image,
					activation=self.activation
					)

	def call(self, inputs):
		assert list(inputs.shape[1:]) == list(self.shape_input), "%s, %s"%(list(inputs.shape[1:]), self.shape_input)
		out = super().call(inputs)
		assert list(out.shape[1:]) == list(self.shape_image), "%s, %s"%(list(out.shape[1:]), list(self.shape_image))
		return out

	def get_config(self):
		return base.convert_config(self._config_param) 

class Decoder64(Decoder):
	def __init__(self, num_latents=10, activation=None, layer_param=None, **kwargs):
		"""This is a decoder that takes in 64x64x3 images
		This is the architecture used in beta-VAE literature
		
		Args:
			num_latents (int): the number of latent elements
			shape_input (list): the shape of the input image (not including batch size)
		"""
		shape_image = [64,64,3]
		if layer_param is None:
			layer_param = ap.decoder64_architecture
		if activation is None:
			activation = ap.decoder64_activations
		super().__init__(
			layer_param=layer_param, 
			num_latents=num_latents, 
			shape_image=shape_image, 
			activation=activation, 
			**kwargs)

class Decoder256(Decoder):
	def __init__(self, num_latents=30, activation=None, layer_param=None, **kwargs):
		"""Decoder network for 512x512x3 images
		
		Args:
		    activation (None, dict): This is a dictionary of specified actions
		"""
		shape_image = [256,256,3]
		if layer_param is None:
			layer_param = ap.decoder256_architecture
		if activation is None:
			activation = ap.decoder256_activations
		super().__init__(
			layer_param=layer_param, 
			num_latents=num_latents, 
			shape_image=shape_image, 
			activation=activation,
			**kwargs)

class Decoder512(Decoder):
	def __init__(self, num_latents=1024, activation=None, layer_param=None, **kwargs):
		"""Decoder network for 512x512x3 images
		
		Args:
		    activation (None, dict): This is a dictionary of specified actions
		"""
		shape_image = [512,512,3]
		if layer_param is None:
			layer_param = ap.decoder512_architecture
		if activation is None:
			activation = ap.decoder512_activations
		super().__init__(
			layer_param=layer_param, 
			num_latents=num_latents, 
			shape_image=shape_image, 
			activation=activation,
			**kwargs)