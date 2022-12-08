# Importing all modules
from matplotlib import pyplot as plt
from pathlib import Path
import argparse
import imutils
import cv2
import numpy as np
from scipy.signal import savgol_filter
import torch
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances

df = pd.read_csv('yolov5/colors.csv')
bgr = df[['b', 'g', 'r']].values

weights='yolov5/runs/train/exp14/weights/best.pt'
model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights)

def get_color(detected_color): 
    color = ""
    if(np.argmax(detected_color) == 0):
        color = "Blue"
    elif(np.argmax(detected_color) == 1):
        color = "Green"
    elif(np.argmax(detected_color) == 2):
        color = "Red"
    else:
        color = "idk"
    return color

#Retrieve image and labels
retrieve = "yolov5/data/images/iPhone/5c.jpg" 
predict = model(retrieve)
img = cv2.imread(retrieve)

xmin, ymin, xmax, ymax = predict.pandas().xyxy[0][['xmin', 'ymin', 'xmax', 'ymax']].astype(int).values[0]
img_crop = img[ymin:ymax, xmin:xmax,:] 
ap = argparse.ArgumentParser()
#ap.add_argument("-i", "--image", required=True,
	#help=retrieve)
#args = vars(ap.parse_args())

# Specifying upper and lower ranges of color to detect in hsv format
lower = np.array([25, 20, 20])
upper = np.array([100, 255, 255])

# split the image into its respective channels, then initialize the
# tuple of channel names along with our figure for plotting
channel = cv2.split(img_crop)
colors = ("b", "g", "r")
plt.figure()
plt.title("'Flattened' Color Histogram")
plt.xlabel("Bins")
plt.ylabel("# of Pixels")

detected_color = []
window = 21
order = 3

import tkinter
import matplotlib
matplotlib.use('TkAgg')

#Loop over color channel and create a histogram for each
for (channel, color) in zip(channel, colors):
    # create a histogram for the current channel and plot it
    hist = cv2.calcHist([channel], [0], None, [256], [0, 256])
    hist_smooth = savgol_filter(hist.T[0], window, order)
    detected_color.append(np.argmax(hist))
    #plt.plot(hist, color=color)
    plt.plot(hist_smooth, '--', color=color)
    plt.xlim([0, 256])

#Plot a 2D color histogram for the
#Green and blue channels
fig = plt.figure()
ax = fig.add_subplot(131)
hist = cv2.calcHist([channel[1], channel[0]], [0, 1], None, [32, 32],
	[0, 256, 0, 256])
p = ax.imshow(hist, interpolation="nearest")
ax.set_title("2D Color Histogram for G and B")
plt.colorbar(p)

#Green and red channels
ax = fig.add_subplot(132)
hist = cv2.calcHist([channel[1], channel[2]], [0, 1], None, [32, 32],
	[0, 256, 0, 256])
p = ax.imshow(hist, interpolation="nearest")
ax.set_title("2D Color Histogram for G and R")
plt.colorbar(p)

#Blue and red channels
ax = fig.add_subplot(133)
hist = cv2.calcHist([channel[0], channel[2]], [0, 1], None, [32, 32],
	[0, 256, 0, 256])
p = ax.imshow(hist, interpolation="nearest")
ax.set_title("2D Color Histogram for B and R")
plt.colorbar(p)

#Generate 2D histogram (two color channels at a time)
print("2D histogram shape: {}, with {} values".format(
	hist.shape, hist.flatten().shape[0]))

#Generate 3D histogram
hist = cv2.calcHist([img], [0, 1, 2],
	None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
print("3D histogram shape: {}, with {} values".format(
	hist.shape, hist.flatten().shape[0]))

print(get_color(detected_color))
print(np.argmax(detected_color))
print(detected_color)
ind = np.argmin(euclidean_distances([detected_color], bgr))

#Display image output
# display the original input image
plt.figure()
plt.axis("off")
plt.imshow(imutils.opencv2matplotlib(img_crop))

blankimage = np.zeros((500, 500, 3), dtype='uint8')
blankimage[:,:,0] = detected_color[2]
blankimage[:,:,1] = detected_color[1]
blankimage[:,:,2] = detected_color[0]
plt.figure()
plt.imshow(blankimage)
plt.title(f"Color: {df.iloc[ind]['name']}")
plt.show()

