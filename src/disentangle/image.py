import tensorflow as tf

@tf.function
def center_square_crop(images, dimensions):
	"""
	center crops the images to a square, with int dimensions.

	Examples:
		>>> a = np.random.normal(size=(3,140,140,3))
		>>> b = np.random.normal(size=(3,140,140,3))
		>>> res1 = center_square_crop(a,128).numpy()
	"""
	image_crop_size = [dimensions,dimensions]
	image_shape = tf.shape(images)
	images=tf.image.crop_to_bounding_box(images, 
		(image_shape[-3]-image_crop_size[0])//2,
		(image_shape[-2]-image_crop_size[1])//2,
		image_crop_size[0],
		image_crop_size[1],
		)
	return images