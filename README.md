# Object Tracking for Speed Estimation
Object Tracking using OpenCV algorithms and speed estimation of the object.

## object_tracking.py
Script for object tracking.

Args:
- `-v` or `--video`: Path to the input video file.
- `-t` or `--tracker`: Tracker algorithm to be used. Defaults to `csrt` if not specified.
- `-o` or `--output`: Path to the output video file. If specified, the tracking result will be saved as a `.mp4` video. Optional.
- `-d` or `--data`: Path to the output data file. If specified, data will be saved to the file as JSON. Optional.

Example usage:
```
python object_tracking.py --video path/to/my/video.mp4 --tracker csrt --output out.mp4 --data data.json
```

## transform.py
Script for coordinate transformation and store the transformed data.

Args:
- `-i` or `--image`: Projective image to be transformed.
- `-d` or `--data`: The JSON data file generated from `object_tracking.py`. The script will take data from this file and add data in it.
