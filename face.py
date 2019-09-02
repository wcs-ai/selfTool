import numpy as np
from SELF_TOOLS import common
import cv2
import scipy


class discern_face(object):
    def __init__(self,imgs,haar_path):
        self.img_path = imgs
        self.imgs = []
        self.face_data = []
        self.haar_file = haar_path
        #检测失败的图片
        self.err_face = []
    #get every img's coordinate of face;[(x,y,w,h),...]
    def find_face_coordinate(self):
        coordinates = []
        for ig in self.img_path:
            img_data = cv2.imread(ig)
            self.imgs.append(img_data)
            img_gray = cv2.cvtColor(img_data,cv2.COLOR_BGR2GRAY)
            face_descade = cv2.CascadeClassifier(self.haar_file)
            faces = face_descade.detectMultiScale(img_gray,1.2,3)
            coordinates.append(faces)

        return coordinates
    #get face data,arguments:output img size
    def get_face_data(self,size,batch=0):
        cds = self.find_face_coordinate()
        i = 1
        #获取人脸部位数据，并统一大小
        for item,img in zip(cds,self.imgs):
            item_in = ''
            if len(item)>0:
                item_in = item[0]
            else:
                self.err_face.append(i-1)
                continue
            image = img[item_in[1]:item_in[1]+item_in[3],item_in[0]:item_in[0]+item_in[2],:]
            #image = img[69:180,69:180,:]
            image_resize = cv2.resize(np.array(image),size,fx=0.25,fy=0.25,interpolation=cv2.INTER_AREA)

            #如果batch为0就不当成迭代对象使用
            if batch>0:
                yield image_resize
            else:
                self.face_data.append(image_resize)
            i += 1
        #return self.face_data
    def save_face_data(self,path):
        np.save(path,self.face_data)
    #get num of img and err_check index.
    def get_info(self):
        num = len(self.img_path)
        return num,self.err_face


