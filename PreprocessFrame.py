from skimage import transform
import warnings
warnings.filterwarnings('ignore')
def PreprocessFrame(frame):
	return_frame=rgb2gray(frame) #greyscale image
	return_frame=transform.resize(return_frame,[240,240]); #crop image
	return_frame=return_frame/255.0 #normalize colors
	return return_frame
exit()