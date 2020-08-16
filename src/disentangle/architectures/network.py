import tensorflow as tf
import numpy as np
from . import base 
from . import block

class NeuralNetwork(block.NetworkBlock):
	"""Base class for NN, can also be used as a simple dense net with batch norm.
	provides functionality for building a sequential layer and can create 
	more complex hierarchical structure than a standard block. For example, can include options.
	"""
	def __init__(self, *ar, shape_input=None, is_create_sequential=True, **kw):
		"""
		See get_available_layer_types() for a full list of avaible layer options
		
		Args:
			*ar: arguments to base.NetworkBlock
			shape_input: This is the shape of the input, not including the batch size.
			is_create_sequential (bool, optional): Whether to wrap entire thing in sequential layer or not. This will also set the inputs if shape_inputs is specified
			**kw: keyword arguments to base.NetworkBlock
		"""
		self.shape_input = shape_input
		self._is_create_sequential = is_create_sequential	
		super().__init__(*ar, **kw)

	def _build_layers(self, layer_param):
		ret = super()._build_layers(layer_param=layer_param)
		if self._is_create_sequential:
			self._create_sequential()
		return ret

	@classmethod
	def get_available_layer_types(cls):
		"""
		This must be a class function
		This creates a list of classes which will create a layer when initialized with parameters
		This function should be overwritten to add functions. For ones with the same checks, 
		additional_check should be also defined here. 
		"""
		batch_norm = base.BatchNormalization
		dense_obj = base.Dense
		dense_opt_obj = block.create_option_block(dense_obj, batch_norm)
		return [batch_norm, dense_obj, dense_opt_obj]

	def _create_sequential(self):
		layers = self.layers
		if not self.shape_input is None:
			layers = [tf.keras.layers.InputLayer(self.shape_input)]+layers
		self.layers = tf.keras.Sequential(layers)

	def call(self, inputs):
		if self._is_create_sequential:
			pred = self.layers(inputs)
		else:
			pred = inputs
			for layer in self.layers:
				pred = layer(pred)
		return pred

class ConvolutionalNeuralNetwork(NeuralNetwork):
	"""
	Base for CNN which can be used to create:
		- Dense layers
		- Conv2D layers
		- Resnet blocks
	Also performs pooling, batch normalizations and flattening
	"""
	@classmethod
	def get_available_layer_types(cls):
		# add pooling and batch norm options
		pool = base.AveragePooling2D
		batch_norm = base.BatchNormalization
		flatten = base.Flatten

		# base objects
		conv2d_obj = base.Conv2D
		conv2d_obj.default_kw = dict(padding="same") # add default keyword arguments, (default_kw only works with BaseTFWrapper instances)
		conv2d_opt_obj = block.create_option_block(conv2d_obj, pool, batch_norm)
		
		dense_obj = base.Dense
		dense_opt_obj = block.create_option_block(dense_obj, batch_norm, flatten, pool)
		
		# custom resnet with batchnorm in internal intermediate conv layers.
		class resnet_obj(block.ResnetBlock):
			@classmethod
			def get_available_layer_types(cls):
				return super().get_available_layer_types()+[conv2d_opt_obj]

		# resnet with external batchnorm and pooling
		resnet_opt_obj = block.create_option_block(resnet_obj, pool, batch_norm)

		return [pool, batch_norm, conv2d_obj, dense_obj, resnet_obj, conv2d_opt_obj, dense_opt_obj, resnet_opt_obj, flatten]
	
class DeconvolutionalNeuralNetwork(NeuralNetwork):
	"""
	Base for Deconv NN which can be used to create:
		- Dense layers
		- Conv2DTranspose layers
		- Resnet blocks
	Also performs upscaling, batch normalizations and reshaping
	"""
	@classmethod
	def get_available_layer_types(cls):
		# add pooling and batch norm options
		upscale = base.UpSampling2D
		upscale.default_kw = dict(interpolation="nearest")
		batch_norm = base.BatchNormalization
		reshape = base.Reshape

		# base objects
		conv2dtranspose_obj = base.Conv2DTranspose
		conv2dtranspose_obj.default_kw = dict(padding="same") # add default keyword arguments, (default_kw only works with BaseTFWrapper instances)
		conv2dtranspose_opt_obj = block.create_option_block(conv2dtranspose_obj, upscale, batch_norm)
		
		dense_obj = base.Dense
		dense_opt_obj = block.create_option_block(dense_obj, batch_norm, reshape, upscale)
		
		# custom resnet with batchnorm in internal intermediate conv layers.
		class resnet_obj(block.ResnetBlock):
			@classmethod
			def get_available_layer_types(cls):
				return [conv2dtranspose_obj, conv2dtranspose_opt_obj]

		# resnet with external batchnorm and upscaleing
		resnet_opt_obj = block.create_option_block(resnet_obj, upscale, batch_norm)

		return [upscale, batch_norm, conv2dtranspose_obj, dense_obj, resnet_obj, conv2dtranspose_opt_obj, dense_opt_obj, resnet_opt_obj, reshape]
	

