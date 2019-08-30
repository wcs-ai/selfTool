import numpy as np
from SELF_TOOLS import common
import cv2
from PIL import Image

class discern_face(object):
    def __init__(self,imgs,haar_path):
        self.img_path = imgs
        self.imgs = []
        self.face_data = []
        self.haar_file = haar_path
    #get every img's coordinate of face;[(x,y,w,h),...]
    def find_face_coordinate(self):
        coordinates = []
        for ig in self.img_path:
            img_data = cv2.imread(ig)
            self.imgs.append(img_data)
            img_gray = cv2.cvtColor(img_data,cv2.COLOR_BGR2GRAY)
            face_descade = cv2.CascadeClassifier(self.haar_file)
            faces = face_descade.detectMultiScale(img_gray,1.3,5)
            coordinates.append(faces)
        return coordinates
    #get face data
    def get_face_data(self):
        cds = self.find_face_coordinate()

        print(self.imgs[0].shape)
        for item,img in zip(cds,self.imgs):
            image = Image.re
            self.face_data.append(img[item[1]:item[1]+item[3],item[0]:item[0]+item[2]])
        return self.face_data
    def save_face_data(self,path):
        np.save(path)

