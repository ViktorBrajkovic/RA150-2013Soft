#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt

np.random.seed(123)  # for reproducibility

from scipy import ndimage
from keras.models import load_model
from keras import backend as K

def return_value(Y_value):
    for i in xrange(0,Y_value.shape[1]):
        if Y_value[0,i] == 1:
            return i

def exists(ele,eles):
    element=[]
    for e in eles:
        x=ele["x"]
        y=ele["y"]
        x2=e["x"]
        y2=e["y"]
        if(abs(x2-x)<8 and abs(y2-y)<8):
            element=e
    return element

def preklapanje(xc, yc, n, k , xl1 ,yl1 , xl2, yl2):
    rastojanje = abs(k*xc+n-yc)
  #  print rastojanje
    if (xc >= min(xl1, xl2) and xc <= max(xl1, xl2)):
        # print "MOS TI TO"
        if (yc >= min(yl1, yl2) and yc <= max(yl1, yl2)):
        #print "Blizu"
        #print xc , xl1 , xl2
             if rastojanje<20:
                #print"IDEMOOOOOOO"
                if(yc<k*xc+n):
                    #print"SAMO HRABRO"
                    return 1
    return 0


model = load_model('my_model.h5')

path = 'C:/Users/Viktor/Desktop/soft/video-4.avi'

cap = cv2.VideoCapture(path)

suma = 0
elements = []

while(cap.isOpened()):


    ret, frame = cap.read()

    lower = np.array([180, 180, 180])
    upper = np.array([255, 255, 255])

    mask = cv2.inRange(frame, lower, upper)
    res = cv2.bitwise_and(frame, frame, mask=mask)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    kernel = np.ones((3,3),np.uint8)
    erosion = cv2.erode(gray,kernel=kernel,iterations=1)
    edges = cv2.Canny(erosion, 120, 100,apertureSize=5)

    minLineLength = 100
    maxLineGap = 10

    lines = cv2.HoughLinesP(edges,1,np.pi/180,50,np.array([]),minLineLength,maxLineGap)
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(res, (x1, y1), (x2, y2), (255, 9, 0), 2)
            cv2.circle(res, (x1, y1), 18, (0, 255, 0), 1)
            cv2.circle(res, (x2, y2), 18, (0, 255, 0), 1)
            #print "uso ",x1,y1,x2,y2
            k=float((y2-y1))/(x2-x1)
            k=round(k,2)
            n=y1-k*x1
            n=round(n,2)
            xl1=x1
            yl1=y1
            xl2=x2
            yl2=y2
        break

    #print k
    #print n
    #print min(k,n)

    label_im, nb_labels = ndimage.label(res)
    objects = ndimage.find_objects(label_im)



    for i in range(nb_labels):
        loc = objects[i]
        (xc, yc) = ((loc[1].stop + loc[1].start) / 2,
                    (loc[0].stop + loc[0].start) / 2)

        (dxc, dyc) = ((loc[1].stop - loc[1].start),
                      (loc[0].stop - loc[0].start))
        if (yc > 14 and yc < frame.shape[0] - 14 and xc > 14 and xc < frame.shape[1] - 14):
            if (dxc > 10 or dyc > 10):
                if (dxc<50 and dyc<50):
                    #print dxc,dyc
                    cv2.circle(frame, (xc, yc), 18, (255 , 255, 0), 1)
                    broj = res[yc - 14:yc + 14, xc - 14:xc + 14]
                 #   print broj.shape
                    brojgray = cv2.cvtColor(broj, cv2.COLOR_BGR2GRAY)
                    brojslika = brojgray.reshape(1, 1, brojgray.shape[0], brojgray.shape[1])
                    prediction = model.predict(brojslika)
                    #b = sum(prediction * [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])


                    newelement = {"x": xc, "y": yc, "value": return_value(prediction), "dodirnuo": 0}
                    newelement2 = {"x": xc, "y": yc, "value": return_value(prediction), "dodirnuo": 1}

                    tempele = exists(newelement, elements)
                    tempele2 = exists(newelement2, elements)

                    a = preklapanje(xc, yc, n, k, xl1, yl1, xl2, yl2)

                    flag=0

                    for i in xrange(len(elements)):
                        if elements[i] == tempele2:
                            if tempele2["dodirnuo"] == 1:
                                newelement["dodirnuo"] = 1
                                elements[i] = newelement
                                temp = newelement
                                flag = 1
                                break


                    if a==1 and newelement["dodirnuo"]==0:
                        newelement["dodirnuo"] = 1
                        elements.append(newelement)
                        try:
                            suma=suma+newelement["value"]
                        except (RuntimeError, TypeError, NameError ,ValueError , EnvironmentError , ReferenceError):
                            a=a
                        print "Suma je ", suma

    cv2.putText(frame, 'Suma: ' + str(suma), (95, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 90, 255), 1)


    cv2.imshow('frame1', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

            #cv2.imshow('frame2', res)

cap.release()
cv2.destroyAllWindows()
   # cv2.waitKey(0)
    #cv2.destroyAllWindows()


'''plt.imshow(frame, cmap='gray' , interpolation='bicubic')
'plt.show()
'''

