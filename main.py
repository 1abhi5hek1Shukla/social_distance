
# Imports
import sys
import json

import cv2
import torch
import numpy as np

from models.experimental import attempt_load
from utils.torch_utils import select_device
from utils.augmentations import letterbox
from utils.general import non_max_suppression, scale_coords

from social_dist.transformations import *
from social_dist.drawing_utils import *

########################################################################

def run_yolov5_detector(frame_org):
	"""
	Running Yolov5 detector to get the detected result
	"""
	frame = letterbox(frame_org, 640,
		                stride=stride, auto=True)[0]
	frame = frame.transpose((2, 0, 1))[::-1]
	frame = np.ascontiguousarray(frame)
	frame = torch.from_numpy(frame).to(processing_device)

	frame = frame / 255.0
	if len(frame.shape) == 3:
		frame = frame[None]

	results = model(frame, augment=False, visualize=False)[0]
	results = non_max_suppression(results, 0.25, 0.45, None, False, max_det=1000)
	return frame, frame_org, results


def process_results(frame, frame_org, results, annotate=True, getBoundingBox = True):
	"""
	Process results to get annotated 
	image or boundingBox or anything else
	"""
	boundingBoxes = []
	for i, det_org in enumerate(results):
		det = det_org.clone() # So that original doesn't alters
		if len(det):
			det[:, :4] = scale_coords(
			            frame.shape[2:], det[:, :4], frame_org.shape).round()
			for *xyxy, conf, clas in reversed(det):
				c = int(clas)
				xmin, ymin, xmax, ymax, = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
				if names[c] == "person":
					if annotate: cv2.rectangle(frame_org, (xmin, ymin), (xmax, ymax), (255,255,255),3)
					if getBoundingBox: boundingBoxes.append([xmin, ymin, xmax, ymax])
	return frame_org, boundingBoxes

def social_distance_monitor(image, bbs_center, mat):
	birdsEyesCenters = np.array(
		list(
			map(getTransformedPoint, bbs_center, [mat]*len(bbs_center))
			)
		)
	pairWiseDist = getPaiWiseDistance(birdsEyesCenters)
	pairWiseDist[pairWiseDist > 100] = 0

	for i in range(len(pairWiseDist)):
		if pairWiseDist[i].any():
			image = cv2.circle(image, getTransformedPoint(bbs_center[i], mat), 50, (0,0,255), -1)
		else:
			image = cv2.circle(image, getTransformedPoint(bbs_center[i], mat), 50, (0,255,0), -1)

	return image

def run_for_image(image, videoData):
	
	# Runnning detector and result processor
	frame, frame_org, results = run_yolov5_detector(image)
	frame_org, boundingBoxes = process_results(frame, frame_org, results, annotate=False)

	# Get and Plot Bounding box center
	boundingBoxesCenter = getBoundingBoxCenter(boundingBoxes)
	# plot_center(frame_org, boundingBoxesCenter)

	# Image Warping
	mat = getWarpingMatrix(videoData["normalPoint"], videoData["warpedPoint"])
	warpedImage = warpPerspective(frame_org, mat, videoData["warpedShape"])
	
	# Plotting the circles
	overlay = warpedImage.copy() # copying overlay before drawing circles
	warpedImage = social_distance_monitor(warpedImage, boundingBoxesCenter, mat)
	# Overlaying
	alpha = 0.8
	warpedImage = cv2.addWeighted(overlay, alpha, warpedImage, 1-alpha, 0)
	# print(getPaiWiseDistance(boundingBoxesCenter))

	# Inversing the warping
	matInv = getWarpingMatrix(videoData["warpedPoint"], videoData["normalPoint"])
	unWarpedImage = warpPerspective(warpedImage, matInv, videoData["normalShape"])

	return unWarpedImage

def developement_image():
	# Image Path and readin
	pathToImage = "./data/images/pedestrian_frame_1.jpg";
	image = cv2.imread(pathToImage)
	image = run_for_image(image)
	display_image(image)


def developement_video():
	# videoDataPath = "./data/videos_info/pedestrian.json"
	videoDataPath = "./data/videos_info/taiwan_pedestrians.json"
	videoData = json.loads(open(videoDataPath).read())

	pathToVideo = videoData["filePath"]

	cap = cv2.VideoCapture(pathToVideo)
	while cap.isOpened():
		ret, frame = cap.read()
		if not ret: break
		frame = run_for_image(frame, videoData)
		# k = display_image(frame, FRM=1)	
		# cv2.imshow(	"Result", cv2.resize(frame, (frame.shape[1]//2, frame.shape[0]//2)))
		cv2.imshow(	"Result", cv2.resize(frame, (frame.shape[1], frame.shape[0])))
		k = cv2.waitKey(1) & 0xff
		if k == 27:
			 break

def developement():
	# developement_image()
	developement_video()

def main():
	developement()

if __name__ == '__main__':
	# Torch Global Setup
	if torch.cuda.is_available():
	    processing_device = select_device('0')
	else:
	    processing_device = select_device('cpu')


	# model = attempt_load('weights/yolov5s.pt',map_location=processing_device)
	model = attempt_load('weights/yolov5n.pt', map_location=processing_device)

	names = model.module.names if hasattr(
		model, 'module'
		) else model.names
	stride = int(model.stride.max())

	# Main Function
	main()


