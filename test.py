import numpy as np
import cv2

 
def nothing(x):
    pass
 
# Creating a window with black image
cv2.namedWindow('image')
 
# creating trackbars for red color change
cv2.createTrackbar('R_l', 'image', 0, 255, nothing)
 
# creating trackbars for Green color change
cv2.createTrackbar('G_l', 'image', 0, 255, nothing)
 
# creating trackbars for Blue color change
cv2.createTrackbar('B_l', 'image', 0, 255, nothing)

# creating trackbars for red color change
cv2.createTrackbar('R_h', 'image', 0, 255, nothing)
 
# creating trackbars for Green color change
cv2.createTrackbar('G_h', 'image', 0, 255, nothing)
 
# creating trackbars for Blue color change
cv2.createTrackbar('B_h', 'image', 0, 255, nothing)
 
cap = cv2.VideoCapture('Video/bully.mp4')

while(True):
	# show image
	_, img = cap.read()
	background = np.zeros((img.shape), dtype="uint8")
	# for button pressing and changing
	k = cv2.waitKey(1) & 0xFF
	if k == 27:
		break

	# get current positions of all Three trackbars
	r1 = cv2.getTrackbarPos('R_l', 'image')
	g1 = cv2.getTrackbarPos('G_l', 'image')
	b1 = cv2.getTrackbarPos('B_l', 'image')
	r2 = cv2.getTrackbarPos('R_h', 'image')
	g2 = cv2.getTrackbarPos('G_h', 'image')
	b2 = cv2.getTrackbarPos('B_h', 'image')
 
    # display color mixture
	mask = cv2.inRange(img, (0, 0, 0), (255, 150, 255))

	maskScaled = mask.copy() / 255.0
	maskScaled = np.dstack([maskScaled] * 3)
	warpedMultiplied = cv2.multiply(img.astype("float"), maskScaled)
	imageMultiplied = cv2.multiply(background.astype(float), 1.0 - maskScaled)
	output = cv2.add(warpedMultiplied, imageMultiplied)
	output = output.astype("uint8")

	cv2.imshow("IMG", mask)
# close the window
cv2.destroyAllWindows()
