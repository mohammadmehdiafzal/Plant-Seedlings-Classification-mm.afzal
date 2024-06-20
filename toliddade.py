from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np
import glob
import cv2
from PIL import Image
from scipy.misc import imsave
import os

#estefadeh az class ImageDataGenerator
datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.0,
        horizontal_flip=True,
        fill_mode='nearest')

a="F:/proje machin arshad/plant/projeasli/data80/tarin1/Sugar beet/*.jpg"
b='F:/proje machin arshad/plant/projeasli/data80/tarin1/Sugar beet/'
images = [cv2.imread(file) for file in glob.glob(a)]
num=1
for shomar in  range(len(images)):
    name=glob.glob(a)[shomar].split('\\')
    name=name[1].split('.');
    name=name[0];
    path=b+name+'.jpg'
    img = load_img(path)
    x = img_to_array(img)
    x = x.reshape((1,) + x.shape)
    i = 0
    for batch in datagen.flow(x, batch_size=1,save_to_dir='F:/proje machin arshad/mohammadmehdi afzal/Plant Seedlings Classification/toliddata',   save_prefix=str(num),      save_format='jpg'):
         i += 1
         if i > 2:
            num=num+1
            break