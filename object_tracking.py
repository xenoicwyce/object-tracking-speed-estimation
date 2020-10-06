# USAGE
# python opencv_object_tracking.py
# python opencv_object_tracking.py --video dashcam_boston.mp4 --tracker csrt

# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
import argparse
import time
import cv2
import json
import os

# data prototype
data = {
    'time': [],
    'video': '',
    'output': '',
    'fps': 0.0,
    'centroid_x': [],
    'centroid_y': [],
}

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", type=str,
    help="path to input video file")
ap.add_argument("-t", "--tracker", type=str, default="csrt",
    help="OpenCV object tracker type")
ap.add_argument('-o', '--output', type=str,
    help='path to output video file')
ap.add_argument('-d', '--data', type=str,
    help='path to output data file')
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
        "boosting": cv2.TrackerBoosting_create,
        "mil": cv2.TrackerMIL_create,
        "tld": cv2.TrackerTLD_create,
        "medianflow": cv2.TrackerMedianFlow_create,
        "mosse": cv2.TrackerMOSSE_create
    }

    # grab the appropriate object tracker using our dictionary of
    # OpenCV object tracker objects
    tracker = OPENCV_OBJECT_TRACKERS[args["tracker"]]()

# initialize the bounding box coordinates of the object we are going
# to track
initBB = None

# if a video path was not supplied, grab the reference to the web cam
if not args.get("video", False):
    print("[INFO] starting video stream...")
    vs = VideoStream(src=0).start()
    time.sleep(1.0)
    input_fps = 30.0

# otherwise, grab a reference to the video file
else:
    vs = cv2.VideoCapture(args["video"])
    data['video'] = args['video']
    input_fps = vs.get(cv2.CAP_PROP_FPS)

data['fps'] = input_fps

# initialize the FPS throughput estimator
fps = None
frame_buffer = []
pause = True # pause flag

# loop over frames from the video stream
while True:
    # grab the current frame, then handle if we are using a
    # VideoStream or VideoCapture object
    frame = vs.read()
    frame = frame[1] if args.get("video", False) else frame

    # check to see if we have reached the end of the stream
    if frame is None:
        break

    # resize the frame (so we can process it faster) and grab the
    # frame dimensions
    # frame = imutils.resize(frame, width=500)
    (H, W) = frame.shape[:2]

    # check to see if we are currently tracking an object
    if initBB is not None:
        # grab the new bounding box coordinates of the object
        (success, box) = tracker.update(frame)

        # check to see if the tracking was a success
        if success:
            (x, y, w, h) = [int(v) for v in box]
            cv2.rectangle(frame, (x, y), (x + w, y + h),
                (0, 255, 0), 2)
            pos = (x + w/2, y + h/2)
            cv2.circle(frame, (int(pos[0]), int(pos[1])), 3, (0, 255, 0), -1)

            data['centroid_x'].append(pos[0])
            data['centroid_y'].append(pos[1])

            # if not prev_pos:
            #     prev_pos = pos
            # else:
            #     delta = (pos[0] - prev_pos[0], pos[1] - prev_pos[1])
            #     speed = math.sqrt(delta[0]**2 + delta[1]**2)*input_fps
            #     prev_pos = pos
            #     data['speed'].append(speed)

        # update the FPS counter
        fps.update()
        fps.stop()

        # initialize the set of information we'll be displaying on
        # the frame
        info = [
            ("Tracker", args["tracker"]),
            ("Success", "Yes" if success else "No"),
            ("FPS", "{:.2f}".format(fps.fps())),
            ('(X, Y)', '({}, {})'.format(x, y)),
        ]

        # loop over the info tuples and draw them on our frame
        for (i, (k, v)) in enumerate(info):
            text = "{}: {}".format(k, v)
            cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        if args.get('output', False):
            frame_buffer.append(frame)

    # show the output frame
    cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if pause:
        key = cv2.waitKey(0) & 0xFF

    # if the 's' key is selected, we are going to "select" a bounding
    # box to track
    if key == ord("s"):
        # select the bounding box of the object we want to track (make
        # sure you press ENTER or SPACE after selecting the ROI)
        initBB = cv2.selectROI("Frame", frame, fromCenter=False,
            showCrosshair=True)

        # start OpenCV object tracker using the supplied bounding box
        # coordinates, then start the FPS throughput estimator as well
        tracker.init(frame, initBB)
        fps = FPS().start()
        pause = False

    # if the `f` key is pressed, go to the next frame
    elif key == ord('f'):
        continue

    # if the `q` key was pressed, break from the loop
    elif key == ord("q"):
        break

# time series array
data['time'] = [n/input_fps for n in range(len(data['centroid_x']))]

# write the tracked output to a video file specified
if args.get('output', False) and len(frame_buffer) != 0:
    dirname = os.path.dirname(args['output'])
    if not os.path.exists(dirname) and dirname:
        os.makedirs(dirname)
    size = (W, H)
    out = cv2.VideoWriter(args['output'], cv2.VideoWriter_fourcc(*'mp4v'),
                          input_fps, size)

    for frame in frame_buffer:
        out.write(frame)

    out.release()
    data['output'] = args['output']

# if we are using a webcam, release the pointer
if not args.get("video", False):
    vs.stop()

# otherwise, release the file pointer
else:
    vs.release()

# close all windows
cv2.destroyAllWindows()

if args.get('data', False):
    dirname = os.path.dirname(args['data'])
    if not os.path.exists(dirname) and dirname:
        os.makedirs(dirname)
    with open(args['data'], 'w') as f:
        json.dump(data, f)