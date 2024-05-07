import numpy as np
import cv2


def order_points(pts):
    #  pts is a list of 4 points specfting (x,y) cordinates
    
    # what np.zeros does is Return a new array of given shape and type, filled with zeros.

    rect = np.zeros((4,2), dtype='float32')  # this will be the shape of the new array and the type of data that will be stored in the array
    # I allocated memory with the above line as well
    #  the shap will also be a recatngle
    
    
    s = pts.sum(axis=1) #What this line does is it sums up the columns of a NP array along the specfied axis 
    rect[0] = pts[np.argmin(s)] # this finds the top-left point with the smallest x+y value
    rect[2] = pts[np.argmax(s)]  # Identifies the bottom-right point with the largest x + y sum
    
    
    diff = np.diff(pts, axis=1) # takes in the array and calculates the discrete diffrence within the columns
    rect[1] = pts[np.argmin(diff)]  # Selects the top right point with the smallest x-y coordinates
    rect[3] = pts[np.argmax(diff)]  # Selects the bottom left point with the largest x-y difference
    
    
    # THis then returns the orderd points
    return rect






def four_point_transform(image, pts):
    # image is the current image I will transform and the images are our ROI 
    # which is my points of intreast
    # ordering the points and unpacking them 
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    
    
    #  Computing the maximum distance between bottom right and bottom left x cordinates or the top right and top left x cordinates 
    widthA = np.sqrt(((br[0]-bl[0]) ** 2) + ((br[1] - bl[1]) **2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1]-tl[1]) ** 2))
    #  The width is the largest distance between the bottom right and bottom left cordiates or the top-right and top-left
    maxWidth = max(int(widthA), int(widthB))
    
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    #  the maximum distance between the top-right and bottom-right y-coordinates 
    # or the top-left and bottom-left y-coordinates
    maxHeight = max(int(heightA), int(heightB))
    
    
    
    #  this allows us to construct a new npArray and get a birds eye view of the image
    # I specify the points [0,0] --> origin,[ maxWidth-1, 0] --> top right, [maxwidth-1, maxHeight-1] --> bottom right, [0, maxHeight-1]  -->  bottom left
    dst = np.array([[0,0], 
                    [maxWidth-1, 0],  
                    [maxWidth-1, maxHeight-1],
                    [0, maxHeight-1]], dtype='float32')
    
    
    # This right here is to actually obtain that birds eye view i was talking about 
    # cv2.getPerspectiveTransform this actually allows us to get that imahe which will give us that 3d perspective
    M = cv2.getPerspectiveTransform(rect, dst) # this is our transform matrix
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight)) # this actually preforms the transformation and we also incldie the width and height of the output image
    
    
    return warped


    