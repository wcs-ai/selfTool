import numpy as np
from SELF_TOOLS import common
import cv2

class discern_face(object):
    def __init__(self,imgs):
        self.imgs = imgs
    #get every img's coordinate of face;[(x,y,w,h),...]
    def find_face_coordinate(self):
        coordinates = []
        for ig in self.imgs:
            img_data = cv2.imread(ig)
            img_gray = cv2.cvtColor(img_data,cv2.COLOR_BGR2GRAY)
            face_descade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
            faces = face_descade.detectMultiScale(img_gray,1.3,5)
            coordinates.append(faces)
        return coordinates
    #get face data
    def get_face_data(self):
        cds = self.find_face_coordinate()
