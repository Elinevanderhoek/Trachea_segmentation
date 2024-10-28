import cv2
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import imutils
import cv2

def midpoint(ptA, ptB):
	return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

def calc_diam_area(segment, final):
    # compute the rotated bounding box of the contour
    orig = segment.copy()

    cnts = cv2.findContours(orig.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    # sort the contours from left-to-right and initialize the 'pixels per metric' calibration variable
    (cnts, _) = contours.sort_contours(cnts)
    pixelsPerMetric = None
    box = cv2.minAreaRect(cnts[0])
    box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
    box = np.array(box, dtype="int")

	# order the points in the contour such that they appear in top-left, top-right, bottom-right, and bottom-left order, then draw the outline of the rotated bounding box
    box = perspective.order_points(box)
    cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)
	
	# loop over the original points and draw them
    for (x, y) in box:
        cv2.circle(orig, (int(x), int(y)), 5, (0, 0, 255), -1)
	
    (tl, tr, br, bl) = box
    (tltrX, tltrY) = midpoint(tl, tr)
    (blbrX, blbrY) = midpoint(bl, br)
	
	# compute the midpoint between the top-left and top-right points, followed by the midpoint between the top-righ and bottom-right
    (tlblX, tlblY) = midpoint(tl, bl)
    (trbrX, trbrY) = midpoint(tr, br)
	
    final2 = final.copy()
	# draw the midpoints on the image
    cv2.circle(final2, (int(tltrX), int(tltrY)), 3, (255, 0, 0), -1)
    cv2.circle(final2, (int(blbrX), int(blbrY)), 3, (255, 0, 0), -1)
    cv2.circle(final2, (int(tlblX), int(tlblY)), 3, (255, 0, 0), -1)
    cv2.circle(final2, (int(trbrX), int(trbrY)), 3, (255, 0, 0), -1)
	
	# draw lines between the midpoints
    cv2.line(final2, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),(255, 0, 255), 2)
    cv2.line(final2, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),(255, 0, 255), 2)
	
    dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
    dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
	
    contours2, _ = cv2.findContours(segment, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    perimeter = 0
    for contour in contours2:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)

    #print(f'Euclidean hight distance:', dA)
    #print(f'Euclidean width distance:', dB)
    #print(f'Area',area)
    #print(f'Circumference', perimeter)

    # # show the output image
    # cv2.imshow("Image", final)

    return dA, dB, area, perimeter, final