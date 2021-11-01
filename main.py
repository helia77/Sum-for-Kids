import cv2
from sklearn.externals import joblib
from skimage.feature import hog
import numpy as np
import os
from random import randrange

#import ctypes  # An included library with Python install.

# Show the numbers to the student

num1 = randrange(10)
num2 = randrange(10)
#ctypes.windll.user32.MessageBoxW(0, "Your numbers are " + num1 + " and " + num2, "Problem", 1)

#num1 = 7
#num2 = 9

print('\nYour numbers are: ', num1, ' and ', num2, '. \n')
print('What is the summation?\n')

# Load the classifier
clf = joblib.load("digits_cls.pkl")

# Capture the handwritten number

key = cv2.waitKey(1)
webcam = cv2.VideoCapture(cv2.CAP_DSHOW)
#os.remove("Picture.jpg")
while True:
    try:
        check, frame = webcam.read()
        cv2.imshow("Capturing", frame)
        key = cv2.waitKey(1)
        if key == ord('s'):
            cv2.imwrite(filename='Picture.jpg', img=frame)
            webcam.release()
            cv2.waitKey(1650)
            cv2.destroyAllWindows()
            #print("Image saved!")
            break

        elif key == ord('q'):
            webcam.release()
            print("Camera off.")
            print("Program ended.")
            cv2.destroyAllWindows()
            break

    except(KeyboardInterrupt):
        webcam.release()
        print("Camera off.")
        print("Program ended.")
        cv2.destroyAllWindows()
        break



# Read the input image
im = cv2.imread(os.getcwd() + '\Picture.jpg')
# Convert to grayscale and apply Gaussian filtering
im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
#cv2.imshow('show', im_gray)
#cv2.waitKey(3000)
im_gray = cv2.GaussianBlur(im_gray, (5, 5), 0)
# Threshold the image

ret, im_th = cv2.threshold(im_gray, 80, 255, cv2.THRESH_BINARY_INV)
#im_gray, 90, 255, cv2.THRESH_BINARY_INV
#cv2.imshow('blah', im_th)
#cv2.waitKey(1000)
# Find contours in the image
#ctrs, hier = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
_, ctrs, _= cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# Get rectangles contains each contour
rects = [cv2.boundingRect(ctr) for ctr in ctrs]
num = 0
cnt = 0
# For each rectangular region, calculate HOG features and predict
# the digit using Linear SVM.
for rect in rects:
    # Draw the rectangles
    cv2.rectangle(im, (rect[0]-5, rect[1]-5), (rect[0] + rect[2]+5, rect[1] + rect[3]+5), (57, 79, 69), 2)
    # Make the rectangular region around the digit
    leng = int(rect[3] * 1.6)
    pt1 = int(rect[1] + rect[3] // 2 - leng // 2)
    pt2 = int(rect[0] + rect[2] // 2 - leng // 2)
    roi = im_th[pt1:pt1+leng, pt2:pt2+leng]
    # Resize the image
    roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
    roi = cv2.dilate(roi, (3, 3))

    # Calculate the HOG features
    roi_hog_fd = hog(roi, orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualize=False)
    nbr = clf.predict(np.array([roi_hog_fd], 'float64'))

    #print('number is: ', nbr[0])
    if cnt == 1:
        num = num + nbr[0]
    else:
        num = num + 10*nbr[0]
    cv2.putText(im, str(int(nbr[0])), (rect[0], rect[1]-15),cv2.FONT_HERSHEY_DUPLEX, 2, (0,0,255), 3)
    cnt = cnt + 1

if num > 18:
    tmp = num//10
    num = (num%10)*10 + tmp
print('\nYour answer is: ',num)
print('sum is: ', num1 + num2)
if num == num1+num2:
    print('\nCorrect, Well Done!')
else:
    print('\nNope! Try harder.')

cv2.imshow("Resulting Image with Rectangular ROIs", im)
cv2.waitKey()