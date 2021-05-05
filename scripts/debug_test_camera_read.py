#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  5 13:49:48 2021

@author: brian
"""

from imutils.video import VideoStream
import argparse
import datetime
import imutils
import time
import cv2

# # construct the argument parser and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-v", "--video", help="path to the video file")
# ap.add_argument("-a", "--min-area", type=int, default=500, help="minimum area size")
# args = vars(ap.parse_args())

# if the video argument is None, then we are reading from webcam
# if args.get("video", None) is None
vs = VideoStream(src=0).start()
time.sleep(2.0)

# initialize the first frame in the video stream
firstFrame = None

# loop over the frames of the video
while True:
	# grab the current frame and initialize the occupied/unoccupied
	# text
	frame = vs.read()
	frame = frame[1]

	# if the frame could not be grabbed, then we have reached the end
	# of the video
	if frame is None:
	    break

	# resize the frame, convert it to grayscale, and blur it
	frame = imutils.resize(frame, width=500)
    cv2.imshow("Security Feed", frame)
    key = cv2.waitKey(1) & 0xFF

	if key == ord("q"):
		break

# cleanup the camera and close any open windows
vs.stop()
vs.release()
cv2.destroyAllWindows()