import glob
import csv
from keras.models import load_model
import numpy as np
import cv2




add='F:\proje machin arshad\Plant Seedlings Classification\sample_submission.csv'
a="F:/proje machin arshad/mohammadmehdi afzal/Plant Seedlings Classification/test/*.jpg"
#a ="F:/proje machin arshad/plantnew/data bi hasheye barabar ziyad/test/*.jpg"

classl=['Black-grass',
'Charlock',
'Cleavers',
'Common Chickweed',
'Common wheat',
'Fat Hen',
'Loose Silky-bent',
'Maize',
'Scentless Mayweed',
'Shepherds Purse',
'Small-flowered Cranesbill',
'Sugar beet']

testk = [cv2.imread(file) for file in glob.glob(a)]
test = np.zeros((len(testk), 80, 80, 3))
#test = np.zeros((len(testk), 32, 32, 3))
for shomar in  range(len(testk)):
    test[shomar,:, :, :]=testk[shomar]
test=np.uint8(test)


listlabel = []
with open(add) as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
         listlabel.append(row)

namepic=glob.glob(a);
nametr=[];
for name in namepic:
    phg=name.split("\\")
    phg=phg[1].split(".")
    nametr.append(phg[0])


#model=load_model("F:/proje machin arshad/plantnew/data ba hashiye barabar kam/model2.h5")
#model=load_model("F:/proje machin arshad/plantnew/data ba hashiye barabar kam/model1.h5")
#model=load_model("F:/proje machin arshad/plantnew/data bi hasheye barabar ziyad/modelkam1.h5")
#model=load_model("F:/proje machin arshad/plantnew/data ba hashiye barabar kam/model3.h5")
model=load_model("F:/proje machin arshad/mohammadmehdi afzal/Plant Seedlings Classification/model2.h5")
preds = model.predict_classes(test/255)

for shomar in  range(len(namepic)):
    listlabel[shomar+1][1]=classl[preds[shomar]]


with open('F:\proje machin arshad\mohammadmehdi afzal\Plant Seedlings Classification\sample_submission.csv', 'w', newline="") as f:
    writer = csv.writer(f)
    writer.writerows(listlabel)