import numpy as np
import glob
import cv2
from PIL import Image
from scipy.misc import imsave
import os

#
#a="F:/proje machin arshad/Dog Breed Identification/train/train/*.jpg"
#a="F:/proje machin arshad/Dog Breed Identification/test/*.jpg"
a="F:/proje machin arshad/plant/test/*.png"
images = [cv2.imread(file) for file in glob.glob(a)]
#images=np.uint8(images);
#num=1
for shomar in  range(len(images)):
#    name=str(num)
    name=glob.glob(a)[shomar].split('\\')
    name=name[1].split('.');
#    name=name[0].split('_');
    name=name[0]
#    img = cv2.cvtColor(images[shomar], cv2.COLOR_BGR2GRAY)
    img = cv2.resize(images[shomar],(80,80))
#    rows,cols = img.shape[:2]
#    M = cv2.getRotationMatrix2D((cols/2,rows/2),30,1)
#    img = cv2.warpAffine(img,M,(cols,rows))
#    path = "F:/proje machin arshad/Dog Breed Identification/poroje/pictrain/"
    path = "F:/proje machin arshad/plantnew/data ba hashiye barabar kam/test/"
    path=path+name+'.jpg'
    cv2.imwrite(path, img)
#    num=num+1