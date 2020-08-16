import inspect
import numpy as np
import tensorflow as tf
from tensorflow.python.training.tracking.data_structures import _DictWrapper as DictWrapper
from tensorflow.python.training.tracking.data_structures import ListWrapper as ListWrapper, NoDependency
from . import base
def convert_dict_key_to_string(d):
	new_d = {}
	for k,v in d.items():
		if type(v) == dict:
			v = convert_dict_key_to_string(v)
		new_d[str(k)] = v
	return new_d


def apply_activation(layer_num, activation, num_layers):
	"""
	gets the correct activation for a given layer number. Starts at layer 0 to n-1
	"""
	if type(activation) == dict or type(activation) == DictWrapper:
		if str(layer_num) in activation:
			activation = activation[str(layer_num)]
		elif str(layer_num - num_layers) in activation:
			activation = activation[str(layer_num - num_layers)]
		else:
			activation = activation["default"]
	else:
		activation = activation
	return activation

class ValidateBlock(base.ValidateParameters):
	@classmethod
	def get_available_layer_types(cls):
		"""
		This must be a class function
		This creates a list of classes which will create a layer when initialized with parameters
		This function should be overwritten to add functions. For ones with the same checks, 
		additional_check should be also defined here. 
		"""
		available_layers_types = []
		return available_layers_types

	@classmethod
	def _check(cls, layer_param, is_check_verbose=False, **kw):
		assert type(is_check_verbose) == bool
		# layer_param can not be empty
		if not layer_param:
			if is_check_verbose: print("layer_param is empty.")
			return False

		# since layer_param[i] has to be loaded in using *layer_param[i], we have to make sure all of the elements in layer_param are lists
		for lp in layer_param:
			if not (type(lp) == ListWrapper or type(lp) == tuple or type(lp) == list):
				if is_check_verbose: print("all of the elements in layer_param have to be convertable to lists, eg. tuple, list, ListWrapper")
				return False

		# checks valid layer type
		if not cls.get_layer_type(layer_param=layer_param, is_check_verbose=is_check_verbose): return False
		
		return True

	@classmethod
	def get_layer_type(cls, layer_param, is_check_verbose=False, **kw):
		# checks all the build layers 
		LayerObjs = cls.get_available_layer_types()
		layer_types = []
		for layer in layer_param:
			correct_layer_type = []
			for i, LayerObj in enumerate(LayerObjs):
				if LayerObj.check(layer_param=layer):
					correct_layer_type.append(i)
			if not len(correct_layer_type) == 1: 
				if is_check_verbose: print("\nThere must be one matching LayerObj but there is %d available layers for layer parameter \"%s\". Available layers:"%(len(
					correct_layer_type), layer), *[LayerObjs[i] for i in correct_layer_type] or ["None"],
				"\n\tPlease specify additional_check for handling of this layer type: %s\n"%(cls.__name__))
				return False # conflicting or nonexistent layer type
			layer_types.append(correct_layer_type[0])
		return layer_types


class NetworkBlock(tf.keras.layers.Layer, ValidateBlock):
	"""provides basic check for network and activation
	"""
	def __init__(self, *layer_param, activation=None, **kw):
		super().__init__()
		assert not activation is None, "activation must be specified"
		if type(activation) == dict: # we need to convert the key to a string as tf backend doesn't accept int keys
			activation = convert_dict_key_to_string(activation)
		# perform layer and activation checks
		assert self.check(layer_param=layer_param, activation=activation, is_check_verbose=True), "Check failed, see last printed message"
		self.activation = activation
		self.num_layers = len(layer_param)
		self.layer_param = layer_param
		self._build_layers(layer_param=layer_param)

	def _build_layers(self, layer_param):
		# assumes check was run
		available_layers_types = self.get_available_layer_types()
		if layer_param == []:
			return
		layer_types = self.get_layer_type(layer_param)
		assert layer_types, "internal error, check must be run before this build"

		# create the layers
		self.layers = []
		for i, lp, lt in zip(range(self.num_layers), layer_param, layer_types):
			layer = available_layers_types[lt](
				*lp, activation=self.apply_activation(i))
			self.layers.append(layer)
	
	@classmethod
	def _check(cls, layer_param, is_check_verbose=False, **kw):
		# check activation if activation are specified
		if not super()._check(layer_param=layer_param, is_check_verbose=is_check_verbose,**kw): return False
		layer_types = cls.get_layer_type(layer_param=layer_param, is_check_verbose=is_check_verbose)
		if "activation" in kw:
			activation = kw["activation"]
			if type(activation) == dict or type(activation) == DictWrapper:
				if not "default" in activation: 
					if is_check_verbose: print("key, \"default\" must be in activation dictionary. Current activation dictionary is", activation)
					return False
			
			elif not callable(activation):
				if is_check_verbose: print(
					"activation must be a dictionary of activation functions or a single activation but is ", 
					activation,
					"of type",
					type(activation))
				return False

			# only NetworkBlock can have dict type activations
			available_lt = cls.get_available_layer_types()
			for i,lt in enumerate(layer_types):
				ancestors = base.get_ancestors(available_lt[lt])
				lt_activation = apply_activation(layer_num=i, activation=activation, num_layers=len(layer_param))
				if (not NetworkBlock in ancestors) and (type(lt_activation) == dict or type(lt_activation) == DictWrapper):
					if is_check_verbose: print("only NetworkBlock can have dict type activations but %s activation is used for %s which has ancestors %s"%(
						dict(lt_activation), available_lt[lt], ancestors))
					return False
		return True


	def apply_activation(self, layer_num):
		"""
		gets the correct activation for a given layer number. Starts at layer 0 to n-1
		"""
		return apply_activation(
			layer_num=layer_num,
			activation=self.activation,
			num_layers=self.num_layers)
	
	def call(self, inputs):
		pred = inputs
		for layer in self.layers:
			pred = layer(pred)
		return pred


# these blocks do not contain any additional options, only the base functions
class ConvBlock(NetworkBlock):
	"""
	Creates a convolutional block given the parameters. 
	Convolutional layer specifications must be [depth, kernel, stride] 

	Activation must be a tensorflow activation function object, which will be the default 
	activation for all of the layers, or a dict
	with layer_number:tensorflow activation function. The dict must constain a "default"
	key, which will be used as the default activation, if not otherwise specified.

	Examples:
		>>> import numpy as np
		>>> inputs = np.random.normal(size=[32,64,64,3])
		>>> activation = tf.nn.relu #{"default":tf.nn.relu, -1:tf.math.sigmoid}
		>>> a = ConvBlock( 
		>>> 	[32,1,1], # layer 1, for matching dimensions 
		>>> 	[32,3,1], # layer 2
		>>> 	[32,3,1], # layer 3
		>>> 	activation = activation
		>>> 	)
		>>> a(inputs) # the model must be run for keras to collect the trainable variables/weights
		>>> print(len(a.weights))

	Args:
		activation: This is the activation functions that will be performed on the network
		*layer_param: convolutional layers specifications in order.
	"""
	@classmethod
	def get_available_layer_types(cls):
		conv2d_obj = base.Conv2D
		conv2d_obj.default_kw = dict(padding="same") # add default keyword arguments, (default_kw only works with BaseTFWrapper instances)
		return [conv2d_obj]

class ResnetBlock(ConvBlock):
	"""
	Creates a convolutional resnet block given the parameters. 
	Convolutional layer specifications must be [depth, kernel, stride] 

	Activation must be a tensorflow activation function object, which will be the default 
	activation for all of the layers, or a dict
	with layer_number:tensorflow activation function. The dict must constain a "default"
	key, which will be used as the default activation, if not otherwise specified.

	Args:
		activation: This is the activation functions that will be performed on the network
		*layer_params: convolutional layers specifications in order.
	"""
	def call(self, inputs):
		pred = inputs
		for i, layer in enumerate(self.layers):
			pred = layer(pred)
			if not i:
				reshaped_initial = pred

		if tuple(pred.shape) == tuple(inputs.shape):
			pred = pred + inputs
		else:
			if not self.apply_activation(0) == tf.keras.activations.linear:
				print(
					'\033[93m'+
					"WARNING: ResNet block is using linear projected layer that is not tf.keras.activations.linear"+
					" and is "+
					str(self.apply_activation(0))+
					" from activations dict: "+
					str(self.activation)+
					" "+
					'\033[0m')
			pred = pred + reshaped_initial

		return pred

# this creates the the options
def create_option_block(base_obj, *ar):
	# will generate a base_obj network block.
	# other than base_obj, all others must inhereit OptionWrapper
	# ar must be additional options
	for ar_obj in ar:
		ancestors = base.get_ancestors(ar_obj)
		assert base.OptionWrapper in ancestors, "other than base_obj, all others must inhereit OptionWrapper"

	class OptionNetworkBlock(NetworkBlock):
		@classmethod
		def get_available_layer_types(cls):
			"""
			This must be a class function
			This creates a list of classes which will create a layer when initialized with parameters
			This function should be overwritten to add functions. For ones with the same checks, 
			additional_check should be also defined here. 
			"""
			return [base_obj, *ar]
		@classmethod
		def _check(cls, layer_param, is_check_verbose=False, **kw):
			if not super()._check(layer_param=layer_param, is_check_verbose=is_check_verbose, **kw):
				return False

			layer_types = cls.get_layer_type(layer_param=layer_param)
			num_base = 0
			for i,l in enumerate(layer_types):
				num_base+=int(not bool(l))
				if num_base>1: #only one base, which should be the 0th available layer and the rest must be options.
					return False

			return True

	return OptionNetworkBlock


	