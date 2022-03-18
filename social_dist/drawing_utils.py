
import cv2

def display_image(image, window = None, FRM=0):
	"""
	Displays an image 
	"""
	image = cv2.resize(image, (image.shape[1]//2, image.shape[0]//2))
	if not window: window = "Frame"
	cv2.imshow(window, image)
	k = cv2.waitKey(FRM) & 0xff
	return k


def plot_center(frame_org, bbs_center):
	"""
	Used for plotting points in the frame (no copy)
	"""
	for center in bbs_center:
		image = cv2.circle(frame_org, tuple(center), 10, (0,0,255), -1)
