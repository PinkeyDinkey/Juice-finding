import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from skimage.io import imread, imshow
#template matching
#Image and template reading
img1 = cv.imread('Data/template_0_0.jpg')
template = cv.imread('Data/train_0.jpg')
h,w,l = template.shape

#Square difference matching is different from the other two in the search process in the upper left corner
res = cv.matchTemplate(img1,template,cv.TM_SQDIFF)
#Result matrix display
plt.imshow(res,cmap=plt.cm.gray)
#Find the best matching position
min_val,max_val,min_loc,max_loc = cv.minMaxLoc(res)
#Find the minimum value in the upper left corner
top_left = min_loc
h,w = template.shape[:2]
#Lower right corner of calculation box
bottom_right=(top_left[0]+w,top_left[1]+h)
#The last three-dimensional vector represents the box color
cv.rectangle(img1,top_left,bottom_right,(255,100,0))
imshow(img1[:,:,::-1])
plt.imshow()
#
#
# #Correlation matching
# res = cv.matchTemplate(img1,template,cv.TM_CCORR)
# #Result matrix display
# plt.imshow(res,cmap=plt.cm.gray)
# #Find the best matching position
# min_val,max_val,min_loc,max_loc = cv.minMaxLoc(res)
# #Find the maximum value in the upper left corner
# top_left = max_loc
# h,w = template.shape[:2]
# #Lower right corner of calculation box
# bottom_right=(top_left[0]+w,top_left[1]+h)
# #The last three-dimensional vector represents the box color
# cv.rectangle(img1,top_left,bottom_right,(255,100,0))
# plt.imshow(img1[:,:,::-1])
#
# #Correlation coefficient matching is similar to the correlation matching process
# res = cv.matchTemplate(img1,template,cv.TM_CCOEFF)
# #Result matrix display
# plt.imshow(res,cmap=plt.cm.gray)
# #Find the best matching position
# min_val,max_val,min_loc,max_loc = cv.minMaxLoc(res)
# #Find the maximum value in the upper left corner
# top_left = max_loc
# h,w = template.shape[:2]
# #Lower right corner of calculation box
# bottom_right=(top_left[0]+w,top_left[1]+h)
# #The last three-dimensional vector represents the box color
# cv.rectangle(img1,top_left,bottom_right,(255,100,0))
# plt.imshow(img1[:,:,::-1])
#
# #Hough line transform (binary image, i.e. gray image)
# img2= cv.imread('rili.jpg')
# #Read the image in the form of gray image (binary image)
# gray = cv.cvtColor(img2,cv.COLOR_BGR2GRAY)
# #Canny edge detection
# edges = cv.Canny(gray,50,150)
# plt.imshow(edges,cmap=plt.cm.gray)
# #Hough line transformation
# lines = cv.HoughLines(edges,0.8,np.pi/180,150)
# #Draw the detected line on the image (polar coordinates)
# for line in lines:
#     rho,theta=line[0]
#     a=np.cos(theta)
#     b=np.sin(theta)
#     x0=a*rho
#     y0=b*rho
#     x1=int(x0+1000*(-b))
#     y1=int(y0+1000*(a))
#     x2=int(x0-1000*(-b))
#     y2=int(y0-1000*(a))
# #Similarly, the last three-dimensional vector represents the line color
#     cv.line(img2,(x1,y1),(x2,y2),(0,255,0))
# plt.imshow(img2[:,:,::-1])
#
# #Hoff circle transformation
# # 1 read the image and convert it to grayscale image
# planets = cv.imread("planet.png")
# gay_img = cv.cvtColor(planets, cv.COLOR_BGRA2GRAY)
# # 2. Blur the median and remove noise
# img3 = cv.medianBlur(gay_img, 7)
# plt.imshow(img3,cmap=plt.cm.gray)
# # 3 Hoff circle detection
# circles = cv.HoughCircles(img3, cv.HOUGH_GRADIENT, 1, 200, param1=100, param2=50, minRadius=0, maxRadius=100)
# #Traverse the image (where the center and radius of the circle must be shaping data)
# for i in circles[0,:]:
#     x0=int(i[0]+0.5)
#     y0=int(i[1]+0.5)
#     r=int(i[2]+0.5)
# #The penultimate 3D variable represents the circle color, - 1 represents the fill
#     cv.circle(planets,(x0,y0),r,(255,0,255),2)
#     cv.circle(planets,(x0,y0),2,(0,255,255),-1)
# plt.imshow(planets[:,:,::-1])