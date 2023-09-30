"""
This script loads a pretrained model from Tensorflow and performs detections
on live camera feed from a connected device (default is integrated webcam on port 0).

Call this function as:

	python webcam-detection-opencv -m/--model model_file.pb -p/--pbtxt config.pbtxt -c/--confidence 0.3
"""


# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
	help="path to Tensorflow model")
ap.add_argument("-p", "--pbtxt", required=True,
	help="path to Tensorflow config file")
ap.add_argument("-c", "--confidence", type=float, default=0.3,
	help="detection threshold; default is 0.3")
args = vars(ap.parse_args())

# Labels for detected classes
CLASSES = ["Background", "Cow"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# Load model using OpenCV
print("[INFO] Loading model...")
net = cv2.dnn.readNetFromTensorflow(args["model"], args["pbtxt"])

# Init video streaming and FPS counter
print("[INFO] Starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)
fps = FPS().start()

# Loop over frames
while True:

	# Grab the frame from the threaded video stream and resize it
	# to have a maximum width of 300 pixels
	frame = vs.read()
	frame = imutils.resize(frame, width=300)

	# Convert frame to blob format; needed for net input
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
		0.007843, (300, 300), 127.5)

	# Net forward prediction
	net.setInput(blob)
	detections = net.forward()

	# For all detections
	for i in np.arange(0, detections.shape[2]):

		# Prediction confidence
		confidence = detections[0, 0, i, 2]

		# Set threshold
		if confidence > args["confidence"]:

			# Class label index (in the cow case it's always 0) and 
			# obtain bounding box dimensions
			idx = int(detections[0, 0, i, 1])
			print(idx)
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# Draw rectangle and add label
			label = "{}: {:.2f}%".format(CLASSES[idx],
				confidence * 100)
			cv2.rectangle(frame, (startX, startY), (endX, endY),
				COLORS[idx], 2)
			y = startY - 15 if startY - 15 > 15 else startY + 15
			cv2.putText(frame, label, (startX, y),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
	
	# Show the video frames
	cv2.imshow("Frame", frame)

	# Set "q" key as breaking point for detection
	if cv2.waitKey(1) == ord("q"):
		break

	# Update the FPS counter
	fps.update()

# Stop timer and display FPS
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# Cleaning open processes
cv2.destroyAllWindows()
vs.stop()