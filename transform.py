import cv2
import json
import yaml
import argparse
import numpy as np

## load config
cfg_file = 'config/transform_config.yaml'

with open(cfg_file) as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)
    hshift = cfg['hshift']
    vsize = cfg['vsize']
    cp_dst = cfg['cp']
##

cp_src = []
def select_point_img(event, x, y, flags, param):
    global cp_src, img_copy
    if event == cv2.EVENT_LBUTTONUP:
        cp_src.append([x, y])
        cv2.circle(img_copy, (x, y), 3, (0, 0, 255), -1)
        print([x, y])

## main
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', type=str,
                help='path to input image file')
ap.add_argument('-d', '--data', type=str,
                help='path to the data json file')
args = vars(ap.parse_args())

with open(args['data']) as f:
    data = json.load(f)

img = cv2.imread(args['image'])
img_copy = img.copy()
cv2.namedWindow('image', flags=cv2.WINDOW_GUI_EXPANDED)
# cv2.setWindowProperty('image', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
cv2.setMouseCallback('image', select_point_img)
while True:
    cv2.imshow('image', img_copy)
    # press `q` when finish selecting the points
    if cv2.waitKey(20) == ord('q'):
        break
cv2.destroyAllWindows()

cp_src = np.float32(cp_src)
cp_dst = np.float32(cp_dst)
cp_dst -= np.float32([hshift, 0])
cp_dst[:, 1] = vsize - cp_dst[:, 1]

M = cv2.getPerspectiveTransform(cp_src, cp_dst)

# transformation to real coordinates
x = np.array(data['centroid_x'])
y = np.array(data['centroid_y'])
nframes = x.shape[0]
coord_with_ones = np.vstack((x, y, np.ones((nframes,))))
coord_tf = M @ coord_with_ones
coord_tf = np.divide(coord_tf[:2, :], coord_tf[2, :])

# compute speed
v = np.diff(coord_tf, axis=1) * data['fps']
speed = np.linalg.norm(v, axis=0)

# write data in file
data['M'] = M.tolist()
data['tformed_x'] = coord_tf[0, :].tolist()
data['tformed_y'] = coord_tf[1, :].tolist()
data['vx'] = v[0, :].tolist()
data['vy'] = v[1, :].tolist()
data['speed'] = speed.tolist()

with open(args['data'], 'w') as f:
    json.dump(data, f)
