import keras
from keras.models import Sequential
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras import regularizers
from keras.callbacks import LearningRateScheduler
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
import numpy as np
import glob
import cv2
import random
from sklearn.model_selection import train_test_split

a1 = "F:/proje machin arshad/mohammadmehdi afzal/Plant Seedlings Classification/tarin1/Black-grass/*.jpg"
a2 = "F:/proje machin arshad/mohammadmehdi afzal/Plant Seedlings Classification/tarin1/Charlock/*.jpg"
a3 = "F:/proje machin arshad/mohammadmehdi afzal/Plant Seedlings Classification/tarin1/Cleavers/*.jpg"
a4 = "F:/proje machin arshad/mohammadmehdi afzal/Plant Seedlings Classification/tarin1/Common Chickweed/*.jpg"
a5 = "F:/proje machin arshad/mohammadmehdi afzal/Plant Seedlings Classification/tarin1/Common wheat/*.jpg"
a6 = "F:/proje machin arshad/mohammadmehdi afzal/Plant Seedlings Classification/tarin1/Fat Hen/*.jpg"
a7 = "F:/proje machin arshad/mohammadmehdi afzal/Plant Seedlings Classification/tarin1/Loose Silky-bent/*.jpg"
a8 = "F:/proje machin arshad/mohammadmehdi afzal/Plant Seedlings Classification/tarin1/Maize/*.jpg"
a9 = "F:/proje machin arshad/mohammadmehdi afzal/Plant Seedlings Classification/tarin1/Scentless Mayweed/*.jpg"
a10 = "F:/proje machin arshad/mohammadmehdi afzal/Plant Seedlings Classification/tarin1/Shepherds Purse/*.jpg"
a11 = "F:/proje machin arshad/mohammadmehdi afzal/Plant Seedlings Classification/tarin1/Small-flowered Cranesbill/*.jpg"
a12 = "F:/proje machin arshad/mohammadmehdi afzal/Plant Seedlings Classification/tarin1/Sugar beet/*.jpg"
at = "F:/proje machin arshad/mohammadmehdi afzal/Plant Seedlings Classification/test/*.jpg"


imagestr1 = [cv2.imread(file) for file in glob.glob(a1)]
imagestr2 = [cv2.imread(file) for file in glob.glob(a2)]
imagestr3 = [cv2.imread(file) for file in glob.glob(a3)]
imagestr4 = [cv2.imread(file) for file in glob.glob(a4)]
imagestr5 = [cv2.imread(file) for file in glob.glob(a5)]
imagestr6 = [cv2.imread(file) for file in glob.glob(a6)]
imagestr7 = [cv2.imread(file) for file in glob.glob(a7)]
imagestr8 = [cv2.imread(file) for file in glob.glob(a8)]
imagestr9 = [cv2.imread(file) for file in glob.glob(a9)]
imagestr10 = [cv2.imread(file) for file in glob.glob(a10)]
imagestr11 = [cv2.imread(file) for file in glob.glob(a11)]
imagestr12 = [cv2.imread(file) for file in glob.glob(a12)]
testk = [cv2.imread(file) for file in glob.glob(at)]

target1 = np.zeros((len(imagestr1),1))
target2 = np.zeros((len(imagestr2),1))
target3 = np.zeros((len(imagestr3),1))
target4 = np.zeros((len(imagestr4),1))
target5 = np.zeros((len(imagestr5),1))
target6 = np.zeros((len(imagestr6),1))
target7 = np.zeros((len(imagestr7),1))
target8 = np.zeros((len(imagestr8),1))
target9 = np.zeros((len(imagestr9),1))
target10 = np.zeros((len(imagestr10),1))
target11 = np.zeros((len(imagestr11),1))
target12 = np.zeros((len(imagestr12),1))


for shomar in range(len(imagestr1)):
    target1[shomar]=0
for shomar in range(len(imagestr2)):
    target2[shomar]=1
for shomar in range(len(imagestr3)):
    target3[shomar]=2
for shomar in range(len(imagestr4)):
    target4[shomar]=3
for shomar in range(len(imagestr5)):
    target5[shomar]=4
for shomar in range(len(imagestr6)):
    target6[shomar]=5
for shomar in range(len(imagestr7)):
    target7[shomar]=6
for shomar in range(len(imagestr8)):
    target8[shomar]=7
for shomar in range(len(imagestr9)):
    target9[shomar]=8
for shomar in range(len(imagestr10)):
    target10[shomar]=9
for shomar in range(len(imagestr11)):
    target11[shomar]=10
for shomar in range(len(imagestr12)):
    target12[shomar]=11


traink=imagestr1+imagestr2+imagestr3+imagestr4+imagestr5+imagestr6+imagestr7+imagestr8+imagestr9+imagestr10+imagestr11+imagestr12

labelk=np.concatenate((target1,target2,target3,target4,target5,target6,target7,target8,target9,target10,target11,target12))

train = np.zeros((len(traink), 80, 80, 3))
test = np.zeros((len(testk), 80, 80, 3))
label = np.zeros((len(traink), 1))
shoam = random.sample(range(len(traink)), len(traink))

for shomar in  range(len(shoam)):
    train[shomar,:, :, :]=traink[shoam[shomar]]
    label[shomar] = labelk[shoam[shomar]]
for shomar in  range(len(testk)):
    test[shomar,:, :, :]=testk[shomar]

train=np.uint8(train);
label=np.uint8(label);
test=np.uint8(test);

X_train, X_valid, Y_train, Y_valid=train_test_split(train,label,test_size=0.2, random_state=12)

model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(80, 80, 3)))
model.add(BatchNormalization())
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.15))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.15))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.15))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.25))
#model.add(Dense(128, activation='relu'))
#model.add(BatchNormalization())
#model.add(Dropout(0.25))
#model.add(Dense(256, activation='relu'))
#model.add(BatchNormalization())
#model.add(Dropout(0.25))
model.add(Dense(12, activation='softmax'))
model.summary()
model.compile(loss='categorical_crossentropy',optimizer=Adam(lr=0.0001, decay=1e-6),metrics=['accuracy'])
    # Train the model
model.fit(X_train / 255 , to_categorical(Y_train),batch_size=128,shuffle=True,epochs=15,validation_data=(X_valid /255, to_categorical(Y_valid)),callbacks=[EarlyStopping(min_delta=0.001, patience=3)])
scores = model.evaluate(X_valid / 255, to_categorical(Y_valid))

print('Loss: %.3f' % scores[0])
print('Accuracy: %.3f' % scores[1])
preds = model.predict_classes(test/255)