import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from tqdm import tqdm
import json
import pickle


X = []
y = []
Y = []
training_data = []
list_of_labels = []
CATEGORIES = ["Pen"]
DATADIR = "E:\LiveED"
json_file = 'Pen_Detector.json'
with open(json_file) as file1:
    lis=[]
    for i in file1:
        lis.append(json.loads(i))
for i in lis:
    list_of_labels.append(i['annotation'][0]['points'][0])




def create_training_data():
    for category in CATEGORIES:  

        path = os.path.join(DATADIR,category)  

        for (img,labels) in zip(tqdm(os.listdir(path)),list_of_labels):  
            try:
                img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)  
                training_data.append([img_array, labels])  
            except Exception as e:  
                continue
create_training_data()
for features, label in training_data:
    X.append(features)
    y.append(label)

X = np.array(X).reshape(-1, 100, 100, 1)
X = X.astype(np.float32)
Y = np.array(y)
Y = Y.astype(np.float32)
print(X[20])
print(Y[20])

cv2.imshow('frame',X[20])
cv2.waitKey(0)


pickle_out = open("X.pickle","wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle","wb")
pickle.dump(Y, pickle_out)
pickle_out.close()





