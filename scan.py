from pyimagesearch.transform import four_point_transform
from skimage.filters import threshold_local # Compute a threshold mask image based on local pixel neighborhood.
import numpy as np
import cv2
import imutils
import argparse



args = argparse.ArgumentParser()
args.add_argument("-i", "--image", required=True, help="The Image we will scan ") # has to be required otherwise we can't parse the image

args = vars(args.parse_args())



'''
I will begin with edge detection here 
'''


# First I will begin loading the image off the disk 
image = cv2.imread(args['image'])



# I will now speed up the image processing portion to make it have a height of 500 pixels
ratio = image.shape[0] / 500.0
orig = image.copy()
image = imutils.resize(image, height=500)

gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) # convert the image from RGB to grayscale
gray = cv2.GaussianBlur() # smoothing the pixel values
edged= cv2.Canny(gray, 75, 200)


# shows the original image and the edge detected image
print("1: Edge Detection")
cv2.imshow("Image", image)
cv2.imshow("Edged", edged)
cv2.waitKey(0) # displaying the window for 
cv2.destroyAllWindows() # Clsoing the windows
