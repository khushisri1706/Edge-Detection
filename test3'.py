import cv2 as cv
import numpy as np
from rembg import remove  
output_path = 'out.jpg'
img = cv.imread("2.jpg")
print(img.shape)
img1 = cv.resize(img, (1280, 720))
cv.imshow("win",img1)

r = cv.selectROI("win", img1)
cropped_image = img1[int(r[1]):int(r[1]+r[3]),
                      int(r[0]):int(r[0]+r[2])]

cv.imwrite("cropped.jpg",cropped_image)
#open(output_path,'wb').write(cropped_image)
input_path = "cropped.jpg"


with open(input_path, 'rb') as i:
    with open(output_path, 'wb') as o:
        input = i.read()
        output = remove(input)
        o.write(output)
image= cv.imread("out.jpg")        
gray= cv.cvtColor(image, cv.COLOR_BGR2GRAY)

edges= cv.Canny(gray,400,600, apertureSize=3)

contours, hierarchy= cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

cv.drawContours(image, contours, -1, (0,255,0),2)
cv.imshow('External Contours', image) 
cv.waitKey(0)
    