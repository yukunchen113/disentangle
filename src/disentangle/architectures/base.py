"""
wraps base tf functions with a check to confirm valid parameters
Configures reading of custom parameter input
- also used as interface for higher level architectures
"""
import tensorflow as tf
from tensorflow.python.training.tracking.data_structures import _DictWrapper as DictWrapper
from tensorflow.python.training.tracking.data_structures import ListWrapper as ListWrapper, NoDependency
from functools import wraps
from types import FunctionType, LambdaType, MethodType
import inspect
##################
# Base Utilities #
##################
def convert_config(x):
	if type(x) == ListWrapper or type(x) == list: 
		x = list(x)
		r = range(len(x))
	elif type(x) == DictWrapper or type(x) == dict: 
		x = dict(x)
		r = x.keys()
	elif isinstance(x, (FunctionType, LambdaType, MethodType)):
		return "%s%s"%(x.__name__, str(inspect.signature(x)))
	else:
		return x

	for i in r:
		x[i] = convert_config(x[i]) 
	return x

def get_ancestors(item):
	if inspect.isclass(item):
		ancestors = [par for par in inspect.getmro(item)]
	else:
		ancestors = [par for par in inspect.getmro(item.__class__)]
	return ancestors


class hybridmethod(object):
	# this code is a method that handles self and cls, code is from: https://stackoverflow.com/questions/18078744/python-hybrid-between-regular-method-and-classmethod
	def __init__(self, func):
		self.func = func
	def __get__(self, obj, cls):
		context = obj if obj is not None else cls
		@wraps(self.func)
		def hybrid(*args, **kw):
			return self.func(context, *args, **kw)
		# mimic methods some more
		hybrid.__func__ = hybrid.im_func = self.func
		hybrid.__self__ = hybrid.im_self = context
		return hybrid



class ValidateParameters:
	@hybridmethod
	def check(cls, layer_param, is_check_verbose=False, **kw):
		"""
		check if a given list is for conv2d
		"""
		# setup
		if type(layer_param) == ListWrapper or type(layer_param) == tuple: layer_param = list(layer_param)

		# check parameters
		if not type(layer_param) == list: 
			if is_check_verbose: print("layer_param must be converatble to list but is type %s"%type(layer_param))
			return False

		if not cls._check(layer_param=layer_param, is_check_verbose=is_check_verbose, **kw): 
			if is_check_verbose: print("checks failed")
			return False

		# additional checks
		if cls.additional_check(layer_param=layer_param, is_check_verbose=is_check_verbose, **kw) is False: 
			if is_check_verbose: print("additional checks failed")
			return False

		return True

	@classmethod # these can be classmethod or regular method.
	def _check(cls, layer_param, **kw):
		return True

	@classmethod # these can be classmethod or regular method.
	def additional_check(cls, layer_param, **kw):
		# additional checks will replace the layer_param with a filtered version.
		return layer_param

################################
# Tensorflow Function Wrappers #
################################
class BaseTFWrapper(ValidateParameters):
	def __init__(self, base_func, default_kw = {}, ignore_kw = []):
		self.base_func = base_func
		self.default_kw = default_kw
		self.ignore_kw = ignore_kw

	def is_kwarg_valid(self, kwarg):
		# does a surface level check (so only for this base func)
		sig = inspect.signature(self.base_func)
		has_var_keyword = False
		for param in sig.parameters.values():
			if param.kind == param.VAR_KEYWORD:
				has_var_keyword = True
		# has_var_keyword only counts if base_func is an instance of BaseTFWrapper, since there
		# needs to be an is_kwarg_valid. Also, since tf functions has ** kwargs, we have to stop before there.  
		is_base_tf_wrapper = BaseTFWrapper in get_ancestors(self.base_func)
		return kwarg in sig.parameters or (has_var_keyword and is_base_tf_wrapper)

	def __call__(self, *ar, **kw):
		"""calls the base function.
		kw should contain the other options like padding and activation
		"""
		kw = {**self.default_kw, **kw} # add any default keywords
		kw = {k:v for k,v in kw.items() if self.is_kwarg_valid(k)} # remove non valid keywords (keywords that are not in base func)

		# selectively get the kwargs according to the user
		if self.ignore_kw == "ALL":
			kw = {}
		elif type(self.ignore_kw) == list:
			kw = {k:v for k,v in kw.items() if not k in self.ignore_kw}
		else:
			raise Exception("self.ignore_kw must be list or ALL, but is:", self.ignore_kw)
		

		assert self.check(ar, is_check_verbose=True), "Checks have failed on given parameters %s for %s"%(ar, self.__class__.__name__)
		return self.base_func(*self.additional_check(ar), **kw)

class Conv2DWrapper(BaseTFWrapper):
	@classmethod
	def _check(self, layer_param, is_check_verbose=False, **kw):
		# Conv specific check
		if not len(layer_param) == 3: 
			if is_check_verbose: print("num layer elements must be 3 but is %d"%len(layer_param))
			return False
		# check each element
		for i in layer_param:
			if not type(i) == int: 
				if is_check_verbose: print("type of layer param elements must be int but is %s"%type(i))
				return False
		return True

class DenseWrapper(BaseTFWrapper):
	@classmethod
	def _check(cls, layer_param, is_check_verbose=False, **kw):
		# Dense specific check
		if not len(layer_param) == 1: 
			if is_check_verbose: print("num layer elements must be 1 but is %d"%len(layer_param))
			return False
		if not type(layer_param[0]) == int: 
			if is_check_verbose: print("type of layer param elements must be int but is %s"%type(layer_param[0]))
			return False
		return True

class OptionWrapper(BaseTFWrapper):

	"""This is for layer_params where you want to specify a preknown identifier for identification of layer type
	Warning:
		This will not check if the arguments that come afterwards is correct

	Attributes:
		identifier (str): This is the unique identifier
	"""

	def __init__(self, *ar, identifier=None, **kw):
		assert not identifier is None, "identifier must be specified"
		super().__init__(*ar, **kw)
		self.identifier = identifier

	def _check(self, layer_param, is_check_verbose=False, **kw):
		if ValidateParameters in get_ancestors(self.base_func):
			filtered_layer_params = self.additional_check(layer_param)
			if filtered_layer_params == False: return False
			return self.base_func.check(layer_param=filtered_layer_params, is_check_verbose=is_check_verbose)
		return True

	def additional_check(self, layer_param, is_check_verbose=False, **kw):
		if not layer_param[0] == self.identifier: 
			if is_check_verbose: print("identifier should be %s at the 0th element for layer_param but is %s for layer_param %s"%(self.identifier, layer_param[0], layer_param))
			return False
		return layer_param[1:]



################################
# Tensorflow Wrapped Functions #
################################
# normal wrapper, these can be option wrapped as well 
Conv2D = Conv2DWrapper(tf.keras.layers.Conv2D)
Conv2DTranspose = Conv2DWrapper(tf.keras.layers.Conv2DTranspose)
Dense = DenseWrapper(tf.keras.layers.Dense)

# option wrapper
BatchNormalization = OptionWrapper(tf.keras.layers.BatchNormalization, identifier="bn")
AveragePooling2D = OptionWrapper(tf.keras.layers.AveragePooling2D, identifier="ap")
UpSampling2D = OptionWrapper(tf.keras.layers.UpSampling2D, identifier="up")
Flatten = OptionWrapper(tf.keras.layers.Flatten, identifier="flatten")
Reshape = OptionWrapper(tf.keras.layers.Reshape, identifier="reshape")
