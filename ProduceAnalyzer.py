from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2
import math

# Author: Jacob O'Grady
# Date: May 6, 2021
# Purpose of Program: This program analyzes an image of produce samples to determine their type, size, and mass. Information is displayed to a text file.
# With some modifications, this program can be used to assist farmers to get quick and accurate calculations for a sample of their produce.
# Used OpenCV and documentation from opencv.org and pyimagesearch.com 

# This will be the number of pixels in the reference object
pixel_ratio = None

# Boolean values for the object recognition
isApple = False
isPotato = False
isCoin = False
isRedApple = False
isFujiApple = False

# Get the image we want to use
image = cv2.imread("produceImage.jpg", cv2.IMREAD_COLOR)

# Convert from the default BGR (Blue, Green, Red) to HSV (Hue, Saturation, Value) color mat
hsv_mat = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Blur the mat to make the contours easier to distinguish
hsv_mat = cv2.GaussianBlur(hsv_mat, (9, 9), 0)

# Convert anther BGR mat to grayscale that will be used for the HoughCircles
gray_mat = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


lower_redOrange = np.array([0, 130, 130], np.uint8)
upper_redOrange = np.array([180, 255, 255], np.uint8)
mask_redOrange = cv2.inRange(hsv_mat, lower_redOrange, upper_redOrange)

lower_red = np.array([40, 0, 0], np.uint8)
upper_red = np.array([180, 255, 255], np.uint8)
mask_red = cv2.inRange(hsv_mat, lower_red, upper_red)

lower_brown = np.array([15, 50, 60], np.uint8)
upper_brown = np.array([180, 160, 190], np.uint8)
mask_brown = cv2.inRange(hsv_mat, lower_brown, upper_brown)

lower_gray = np.array([0, 0, 0], np.uint8)
upper_gray = np.array([60, 85, 100], np.uint8)
mask_gray = cv2.inRange(hsv_mat, lower_gray, upper_gray)

colorContours_redO, hierarchy = cv2.findContours(mask_redOrange, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
colorContours_red, hierarchy = cv2.findContours(mask_red, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
colorContours_brown, hierarchy = cv2.findContours(mask_brown, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
colorContours_gray, hierarchy = cv2.findContours(mask_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Reset the text file
f = open("calculations.txt", "w")
f.write("")
f.close()

# -----------------------------------------------------------------------------------------------------------------------------------------
# This section uses the Haar Cascade Classifier to detect apples. 
objectRecognizer = cv2.CascadeClassifier("cascade.xml")
  
foundApple = objectRecognizer.detectMultiScale(gray_mat, minSize =(200, 200))
                                  
numApples = len(foundApple)


if numApples != 0:
      
    for (x, y, width, height) in foundApple:
          
        cv2.rectangle(image, (x, y), (x + height, y + width), (0, 255, 255), 3)
                      
 # -----------------------------------------------------------------------------------------------------------------------------------------
 # This section finds a circle that will be used as the reference image. The circle is of a known size, so it can be used
 # to find the size of the other objects. The object used in this image is a Quarter.                     

detected_circles = cv2.HoughCircles(gray_mat, cv2.HOUGH_GRADIENT, 1, hsv_mat.shape[0], param1=100, param2=40, minRadius=25, maxRadius=40)
  
# Find and draw any circles if any are found 
if detected_circles is not None: 
  
    # Use a np array to convert the circle parameters a, b, and r to integers
    detected_circles = np.uint16(np.around(detected_circles)) 
  
    for point in detected_circles[0]:
        a = point[0]
        b = point[1]
        r = point[2] 
    
        # Draw the circumference of the circle
        cv2.circle(image, (a, b), r, (0, 255, 0), 2) #args: image, center coordinate, radius, color (green in this case), thickness
  
        # Get the number of pixels of the diameter to be used in the calculations
        circle_pixels = r * 2
else:
    
    print("No circles were detected!")

# ------------------------------------------------------------------------------------------------------------------------------------
# This section finds the contours of the objects

# Use Canny edge detection to find the edges
edge_detect = cv2.Canny(hsv_mat, 10, 100) # source image, threshold 1, threshold 2
edge_detect = cv2.dilate(edge_detect, None) 
edge_detect = cv2.erode(edge_detect, None)

# Use the findContours function to find the contours of each object 
foundContours = cv2.findContours(edge_detect.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # args: image, mode, method
foundContours = imutils.grab_contours(foundContours)

# Sort the contours from left to right
(foundContours, _) = contours.sort_contours(foundContours, method="left-to-right")

# -------------------------------------------------------------------------------------------------------------------------------------------
# This secion loops through each contour and performs calculations

# This function finds the midpoints of each line segment
# Formula = [(X1, Y1)/2],[(X2, Y2)/2]
def getMidPoint(x, y):
    return ((x[0] + y[0]) * 0.5, (x[1] + y[1]) * 0.5)

cnt_contours = 0

for contour in foundContours:

    # Ignore contours that are too small
    if cv2.contourArea(contour) < 400: 
        continue

    cnt_contours += 1

    # Get the smallest rectangle around the contours
    boundBox = cv2.minAreaRect(contour)
    boundBox = cv2.boxPoints(boundBox)
    boundBox = np.array(boundBox)

    # Draw the contours within the bounding box
    cv2.drawContours(image, [boundBox.astype("int")], 0, (0, 255, 0), 2)

    # Get the points out of the boundBox
    for (x1, y1) in boundBox:

        # Draw points for each rectangle
        cv2.circle(image, (x1.astype("int"), y1.astype("int")), 3, (255, 0, 0), -1) #agrs: image, center, radius, color (blue), thickness

        # Get the points from the bounding box
        (point1, point2, point3, point4) = boundBox

        # Use the midpoint function to find the points that will be used to calculate length and width
        # The different combinations needed are point1 with point2 x and y, point3 with point4 x and y, point1 with point4 x and y, and point2 with point3 x and y
        (pnt1_pnt2_x, pnt1_pnt2_y) = getMidPoint(point1, point2)
        (pnt3_pnt4_x, pnt3_pnt4_y) = getMidPoint(point3, point4)
        (pnt1_pnt4_x, pnt1_pnt4_y) = getMidPoint(point1, point4)
        (pnt2_pnt3_x, pnt2_pnt3_y) = getMidPoint(point2, point3)

        # Find the center point using the midpoint of two oposite points
        (centerpoint_x, centerpoint_y) = getMidPoint(point1, point3)

        # Convert the center point to the correct type
        centerPoint = (centerpoint_x.astype("int"), centerpoint_y.astype("int"))

        # Draw a circle in the middle just because
        cv2.circle(image, centerPoint, 3, (255, 0, 255), -1) #agrs: image, center, radius, color (blue), thickness

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------
# This section checks if the centerpoint of each contour is within the rectangle of the detected apple

    for (x, y, w, h) in foundApple:

        #top left
        applePoint1 = (x,y)
        #top right
        applePoint2 = (x+w, y)
        #bottom right
        applePoint3 = (x+w, y+h)
        #bottom left
        applePoint4 = (x, y+h)

    # If the center of the contour is between the top left and bottom right point of an apple contour, the object is an apple
    if((centerPoint>applePoint1) & (centerPoint<applePoint3)):
        #print("The centerpoint is within the apple")
        isApple = True
        isPotato = False
    else:
        isApple = False
        isPotato = True

# --------------------------------------------------------------------------------------------------------------------------------------------------------
# This section gets the color information from each object. A select few colors have been picked to be detected. More colors can be added as needed. 

    for pic, contour in enumerate(colorContours_redO):
        
        area = cv2.contourArea(contour)
        
        if(area > 5000):
            x, y, w, h = cv2.boundingRect(contour)
            
            colorPoint1_ro = (x, y)
            colorPoint2_ro = (x + w, y + h)

            image = cv2.rectangle(image, colorPoint1_ro, colorPoint2_ro,(0, 69, 255), 2)
   
            cv2.putText(image, "Orange", colorPoint2_ro, cv2.FONT_HERSHEY_SIMPLEX, 1.0,(0, 69, 255))

            if((centerpoint_x > x) & (centerpoint_x < (x+w)) & (centerpoint_y > y) & (centerpoint_y < (y+w))):
                print("Fuji Apple Found")
                isFujiApple = True
                isCoin = False
                isRedApple = False
                isPotato = False
                            
    for pic, contour in enumerate(colorContours_red):
        area = cv2.contourArea(contour)
        if(area > 5000):
            x, y, w, h = cv2.boundingRect(contour)
            
            # top left
            colorPoint1_red = (x, y)
            # bottom right
            colorPoint2_red = (x + w, y + h)
           
            image = cv2.rectangle(image, colorPoint1_red, colorPoint2_red, (0, 0, 255), 2) 
   
            cv2.putText(image, "Red", colorPoint2_red, cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255))

            if((centerpoint_x > x) & (centerpoint_x < (x+w)) & (centerpoint_y > y) & (centerpoint_y < (y+w))):
                print("Red Apple Found")
                isCoin = False
                isRedApple = True
                isPotato = False
                isFujiApple = False

           
    for pic, contour in enumerate(colorContours_brown):
        area = cv2.contourArea(contour)
        if(area > 5000):
            x, y, w, h = cv2.boundingRect(contour)
            
            colorPoint1_brown = (x, y)
            colorPoint2_brown = (x + w, y + h)

            image = cv2.rectangle(image, colorPoint1_brown, colorPoint2_brown,(0, 30, 58), 2) 
   
            cv2.putText(image, "Brown", colorPoint2_brown, cv2.FONT_HERSHEY_SIMPLEX, 1.0,(0, 30, 58))

            
            #if((centerPoint>colorPoint1_brown) & (centerPoint<colorPoint2_brown)):
            if((centerpoint_x > x) & (centerpoint_x < (x+w)) & (centerpoint_y > y) & (centerpoint_y < (y+w))):
                print("Potato Found")
                isCoin = False
                isPotato = True
                isFujiApple = False
                isRedApple = False


    for pic, contour in enumerate(colorContours_gray):
        area = cv2.contourArea(contour)
        if(area > 500):
            x, y, w, h = cv2.boundingRect(contour)
            
            colorPoint1_gray = (x, y)
            colorPoint2_gray = (x + w, y + h)
            
            image = cv2.rectangle(image, colorPoint1_gray, colorPoint2_gray,(128, 128, 128), 2) 

            cv2.putText(image, "Gray", colorPoint2_gray, cv2.FONT_HERSHEY_SIMPLEX, 1.0,(128, 128, 128)) 

            if((centerpoint_x > x) & (centerpoint_x < (x+w)) & (centerpoint_y > y) & (centerpoint_y < (y+w))):
                print("Coin Found")
                isCoin = True
                isFujiApple = False
                isRedApple = False
                isPotato = False


# ---------------------------------------------------------------------------------------------------------------------------------------------------
# This section performs distance calculations between points

    # Find the Euclidean distances between the calculated midpoints. Length vs width doesn't matter yet, so I just call them distA and distB
    calcDistA = dist.euclidean((pnt1_pnt2_x, pnt1_pnt2_y), (pnt3_pnt4_x, pnt3_pnt4_y))
    calcDistB = dist.euclidean((pnt1_pnt4_x, pnt1_pnt4_y), (pnt2_pnt3_x, pnt2_pnt3_y))
  
    # Use the ratio between the known size of the Quarter and the number of pixels found in the Quarter to determine object size
    # The known size of a Quarter is 2.4 cm
	
    pixel_ratio = circle_pixels / 2.4
 
    distA = calcDistA / pixel_ratio
    
    distB = calcDistB / pixel_ratio

    area = distA * distB

    # Find which distance is greater
    if(distA > distB):
        longDist = distA
        shortDist = distB
    else:
       shortDist = distA
       longDist = distB

    # Equation to find volume of oblate spheroid
    volume = ((4/3) * math.pi * (longDist * longDist) * shortDist)

    #Avg density of potato = 0.63 g/cm^3
    #Avg density of apple = 0.46 g/cm^3
    # Mass = Volume x Density
    potatoMass = volume * 0.63
    appleMass = volume * 0.46

# -------------------------------------------------------------------------------------------------------------------------------------------------------------
# This section displays text on screen and writes the information to a text file

    # Get the number of potatoes. Account for the reference object
    numPotatoes = cnt_contours - numApples - 1


    cv2.putText(image, "{:.2f} cm".format(distB), (int(pnt1_pnt2_x), int(pnt1_pnt2_y)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)
		
		
    cv2.putText(image, "{:.2f} cm".format(distA), (int(pnt2_pnt3_x), int(pnt2_pnt3_y)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)
		
		
    # Write the information to a text file called "calculations.txt"

    if(isCoin == False):
    
        f = open("calculations.txt", "a")
        #f.write(f'\nNumber of Apples: {numApples}\n')
        #f.write(f'Number of Potatoes: {numPotatoes}\n')
        if(isFujiApple):
            f.write('Fuji Apple: \n')
        if(isRedApple):
            f.write('Red Apple: \n')
        if(isPotato):
            f.write('Potato: \n')

        f.write(f'Distance A: {distA} cm\n')
        f.write(f'Distance B: {distB} cm\n')
        f.write(f'Area of Object: {area} cm^2\n')

        if(isApple):
            f.write(f'Apple mass = {appleMass} grams\n\n')

        else:
            f.write(f'Potato mass = {potatoMass} grams\n\n')

        f.close()

    # Display the image
    cv2.imshow("Image", image)
    cv2.waitKey(0)

f = open("calculations.txt", "a")
f.write(f'Number of Apples: {numApples}\n')
f.write(f'Number of Potatos: {numPotatoes}\n')
f.close()


