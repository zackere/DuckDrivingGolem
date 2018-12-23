import skimage
import warnings
import numpy
warnings.filterwarnings('ignore')
def PreprocessFrame(frame):
	return_frame=skimage.color.rgb2gray(frame) #greyscale image
	return_frame=skimage.transform.resize(return_frame,[120,120]) #crop image
	return_frame=numpy.reshape(return_frame,120*120)
	return_frame=numpy.resize(return_frame,[1,120*120])
	return return_frame