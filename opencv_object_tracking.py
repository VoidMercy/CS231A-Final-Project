# USAGE
import os
# python opencv_object_tracking.py
# python opencv_object_tracking.py --video dashcam_boston.mp4 --tracker csrt
# Press `s` key to begin tracking
# CREDITS: https://www.pyimagesearch.com/2018/07/30/opencv-object-tracking/

# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
import argparse
import imutils
import time
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", type=str,
	help="path to input video file")
ap.add_argument("-t", "--tracker", type=str, default="csrt",
	help="OpenCV object tracker type")
args = vars(ap.parse_args())

# extract the OpenCV version info
(major, minor) = cv2.__version__.split(".")[:2]

# if we are using OpenCV 3.2 OR BEFORE, we can use a special factory
# function to create our object tracker
if int(major) == 3 and int(minor) < 3:
	tracker = cv2.Tracker_create(args["tracker"].upper())

# otherwise, for OpenCV 3.3 OR NEWER, we need to explicity call the
# approrpiate object tracker constructor:
else:
	# initialize a dictionary that maps strings to their corresponding
	# OpenCV object tracker implementations
	OPENCV_OBJECT_TRACKERS = {
		"csrt": cv2.TrackerCSRT_create,
		"kcf": cv2.TrackerKCF_create,
		"mil": cv2.TrackerMIL_create,
		# "tld": cv2.TrackerTLD_create,
		# "medianflow": cv2.TrackerMedianFlow_create,
		# "mosse": cv2.TrackerMOSSE_create
	}

	# grab the appropriate object tracker using our dictionary of
	# OpenCV object tracker objects
	p = cv2.TrackerCSRT_Params()
	p.psr_threshold = 0.00
	tracker = OPENCV_OBJECT_TRACKERS[args["tracker"]](p)

# initialize the bounding box coordinates of the object we are going
# to track
initBB = None

# if a video path was not supplied, grab the reference to the web cam
if not args.get("video", False):
	print("[INFO] starting video stream...")
	vs = VideoStream(src=0).start()
	time.sleep(1.0)

# otherwise, grab a reference to the video file
else:
	vs = cv2.VideoCapture(args["video"])

# initialize the FPS throughput estimator
fps = None

initArea = None
initW = 0
initH = 0

count = 0
prevBox = None

out = open("csrt-results.txt", "w")

def dist(a, b):
	return (a[0]-b[0])**2 + (a[1]-b[1])**2

tick = 0
# loop over frames from the video stream
while True:
	# grab the current frame, then handle if we are using a
	# VideoStream or VideoCapture object
	frame = vs.read()
	frame = frame[1] if args.get("video", False) else frame
	frame = frame[116:580, 300:953]

	if initBB is None:
		# select the bounding box of the object we want to track (make
		# sure you press ENTER or SPACE after selecting the ROI)
		# initBB = cv2.selectROI("Frame", frame, fromCenter=False,
			# showCrosshair=True)
		initBB = (265, 204, 125, 125)
		# initBB = (546, 289, 182, 245)
		# initBB = (538, 283, 198, 258)
		prevBox = initBB
		initArea = initBB[2] * initBB[3]
		initW = initBB[2]
		initH = initBB[3]

		# start OpenCV object tracker using the supplied bounding box
		# coordinates, then start the FPS throughput estimator as well
		tracker.init(frame, initBB)
		fps = FPS().start()
		continue

	# check to see if we have reached the end of the stream
	if frame is None:
		break

	# resize the frame (so we can process it faster) and grab the
	# frame dimensions
	# frame = imutils.resize(frame, width=500)
	(H, W) = frame.shape[:2]

	# check to see if we are currently tracking an object
	if initBB is not None:
		box = None

		if box is None:
			# grab the new bounding box coordinates of the object
			(success, box) = tracker.update(frame)
		area = box[2] * box[3]
		# if dist(box, prevBox) <= 10.0:
		# 	count += 1
		# else:
		# 	count = 0
		# if count >= 20:
		# 	if os.path.exists(f"frames/frame{tick}.txt"):
		# 		print("TRUTH")
		# 		data = open(f"frames/frame{tick}.txt", "r").read().strip().split(",")
		# 		data = [float(i) for i in data]
		# 		box = (int(data[0] - initW * 0.45), int(data[1] - initH * 0.45), initW, initH)
		# 		tracker = OPENCV_OBJECT_TRACKERS[args["tracker"]]()
		# 		tracker.init(frame, box)
		# 		count = 0
		# else:
		# 	if area < 0.6 * initArea or area > initArea / 0.75:
		# 		print("REFRESH", box, initArea, area)
		# 		new_box = (box[0] - (initW - box[2]) // 4, box[1] - (initH - box[3]) // 4, box[2] + (initW - box[2]) // 2, box[3] + (initH - box[3]) // 2) 
		# 		tracker = OPENCV_OBJECT_TRACKERS[args["tracker"]]()
		# 		tracker.init(frame, new_box)
		# 		count = 0
		# 	initBB = box

		# check to see if the tracking was a success
		if success:
			(x, y, w, h) = [int(v) for v in box]
			out.write(str((tick, x, y, w, h)) + "\n")
			# cv2.rectangle(frame, (x, y), (x + w, y + h),
			# 	(0, 255, 0), 2)

		# update the FPS counter
		fps.update()
		fps.stop()

		# initialize the set of information we'll be displaying on
		# the frame
		info = [
			("Tracker", args["tracker"]),
			("Success", "Yes" if success else "No"),
			("FPS", "{:.2f}".format(fps.fps())),
		]

		# loop over the info tuples and draw them on our frame
		# for (i, (k, v)) in enumerate(info):
		# 	text = "{}: {}".format(k, v)
		# 	cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
		# 		cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
	# cv2.imshow("Frame", frame)
	(x, y, w, h) = [int(v) for v in initBB]
	# cv2.rectangle(frame, (x, y), (x + w, y + h),
	# 	(0, 255, 0), 2)

	# key = cv2.waitKey(1) & 0xFF
	tick += 1
	if tick % 100 == 0:
		print(tick)
	# if key == ord("q"):
	# 	break
	prevBox = initBB
out.close()
# with open("result.txt", "w") as win:
# 	win.write(str(out))

# if we are using a webcam, release the pointer
if not args.get("video", False):
	vs.stop()

# otherwise, release the file pointer
else:
	vs.release()

# close all windows
cv2.destroyAllWindows()