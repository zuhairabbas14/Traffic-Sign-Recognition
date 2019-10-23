import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2
import csv
import random
import seaborn as sns
import operator
import sklearn
import time
from random import randrange
from collections import Counter
from itertools import cycle
from sklearn.ensemble import RandomForestClassifier


def read_training_data(rootpath):
    
    images = []
    labels = []
    
    for c in range(0,43):
        prefix = rootpath + '/' + format(c, '05d') + '/'
        gtFile = open(prefix + 'GT-'+ format(c, '05d') + '.csv')
        gtReader = csv.reader(gtFile, delimiter=';')
        next(gtReader)
        
        for row in gtReader:
            images.append(plt.imread(prefix + row[0]))
            labels.append(int(row[7]))
        gtFile.close()
        
    return images, labels


def read_test_data(rootpath):
    
    images = []
    labels = []
    prefix = rootpath + '/'
    gtFile = open(prefix + 'GT-final_test.csv')
    gtReader = csv.reader(gtFile, delimiter=';')
    next(gtReader)
    
    for row in gtReader:
        images.append(plt.imread(prefix + row[0]))
        labels.append(int(row[7]))
    gtFile.close()
    
    return images, labels


def add_padding(src):
    
    width = src.shape[0]
    length = src.shape[1]
    borderType = cv2.BORDER_REPLICATE
    top, bottom, left, right = 0,0,0,0

    if width > length:
        diff = width - length
        left = diff
        right = diff
    else: 
        diff = length - width
        top = diff
        bottom = diff

    image = cv2.copyMakeBorder( src, top, bottom, left, right, borderType)
    return image


def resize_image(image, dim):
    
    resized_image = cv2.resize(image, (dim, dim)) 
    return resized_image


def noise(image, temp):
    
    output = np.zeros(image.shape,np.uint8)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            output[i][j] = image[i][j] + random.randint(-temp, temp)
    return output


def rotate(image, angle):
    
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result


def image_augmentation(image, temp, angle):
    
    new_img = noise(image, temp)
    final_img = rotate(new_img, angle)
    return final_img


def image_transformation(images, dim):
    
    fin = []
    for img in images:
        i = add_padding(img)
        i = resize_image(i, dim)
        fin.append(i)
        
    return fin


def train_test_split(X, Y, split, batch_size):

    train_x = list()
    train_y = list()
    test_size = round((1 - split) * len(X))
    val_x = X[:]
    val_y = Y[:]

    while len(val_x) > test_size:
        lst_x = []
        lst_y = []
        index = randrange(batch_size, len(val_x), batch_size)

        for i in range(index - 1, (index - batch_size) - 1, -1):
            lst_x.append(val_x.pop(i))
            lst_y.append(val_y.pop(i))
        train_x.extend(lst_x)
        train_y.extend(lst_y)

    return train_x, train_y, val_x, val_y


def frequency_plot(y_train):
    
    temp_labels = y_train
    temp_labels = sorted(temp_labels)
    temp_labels = dict(Counter(temp_labels))
    plt.figure(figsize=(8,4))
    plt.bar(range(len(temp_labels)), temp_labels.values(), align='center', color='green',label='Training', alpha=0.3, width=0.7)
    plt.xlabel('Classes')
    plt.ylabel('Distribution')
    plt.title('Frequency Distribution of Class Labels',fontsize=14)
    plt.show()


def data_augmentation(batch_size, x_train, y_train, dim):
    
    temp_labels = y_train
    temp_labels = dict(Counter(temp_labels))
    temp_labels = sorted(temp_labels.items(), key=lambda kv: kv[1])
    limit = temp_labels[len(temp_labels)-1][1]
    train_x = x_train
    train_y = np.asarray(y_train)

    for x in temp_labels:
        y = x[0]
        count_y = x[1]
        indices = np.argwhere(train_y==y).flatten()

        while count_y < limit:
            lst_x = []
            lst_y = []
            index = randrange(batch_size, len(indices), batch_size)

            for i in range(index - 1, (index - batch_size) - 1, -1):
                temp = random.randint(14, 25)
                angle = random.randint(0, 25)
                to_augment = np.asarray(train_x[indices[i]]).reshape(dim, dim, 3)
                lst_x.append(image_augmentation(to_augment, temp, angle))
                lst_y.append(train_y[indices[i]])

            count_y += 30
            x_train.extend(lst_x)
            y_train.extend(lst_y)
    
    frequency_plot(y_train)


def normalize_images(x):
    return np.array((x - np.min(x)) / (np.max(x) - np.min(x)))


def ravel_image(data):
    
    images_1d = []
    for img in data:
        y = list(img.ravel())
        images_1d.append(y)
    return images_1d


def main_handler(dim=30, aug=True):
    
    batch_size = 30 # images are in the batch of 30
    # transforming image by adding padding and resizing to desired size
    images, x_test = image_transformation(train_images, dim), image_transformation(test_images, dim)
    # splitting in to train and validation sets
    x_train, y_train, x_val, y_val = train_test_split(images, labels, 0.8, batch_size)
    # bar chart showing classes distribution before augmentation
    frequency_plot(y_train)
    
    if aug == True:
        data_augmentation(batch_size, x_train, y_train, dim)
         
    # normalization of training, test and validation set input features
    x_train, x_test, x_val = normalize_images(x_train), normalize_images(x_test), normalize_images(x_val)
    # ravelling training, test and validation set input features
    x_train, x_test, x_val = ravel_image(x_train), ravel_image(x_test), ravel_image(x_val)
    
    # Building a random forest classifier
    model = RandomForestClassifier(n_estimators = 100, criterion='entropy', min_samples_split=50, max_depth=30)
    t0=time.time()
    model.fit(x_train, y_train)
    train_time = round(time.time()-t0, 3)
#     y_pred_test = model.predict(x_test)
#     report = sklearn.metrics.classification_report(y_true= y_test, y_pred= y_pred_test)
#     print(report)
    acc_val = model.score(x_val, y_val)
    acc_test = model.score(x_test, y_test)
    print('Validation Accuracy: ' + str(acc_val))
    print('Test Accuracy: ' + str(acc_test))
    print('Training time: ' + str(train_time) + 's')
    return acc_test, train_time


# os.chdir('D://')
train_images, labels = read_training_data('./GTSRB/Final_Training/Images')
test_images, y_test = read_test_data('./GTSRB/Final_Test/Images')
print("<> Without DATA Augmentation <>")
main_handler(30, False)
print("-----------------------------\n")

dimensions = [15, 20, 25, 30, 40]
results = {}

for dim in dimensions:
    print("<> With size: " + str(dim) + ' x ' + str(dim))
    results[str(dim)] = main_handler(dim, True)
    print("-----------------------------\n")
print(results)

d = [x+'x'+x for x in list(results.keys())]
t = [x[1] for x in list(results.values())]
a = [(x[0]*100) for x in list(results.values())]

sns.set()
plt.plot(d, t) 
plt.xlabel('Image Size') 
plt.ylabel('Time (s)') 
plt.title('Time Vs Image Size') 
plt.show() 

plt.figure()
plt.plot(d, a) 
plt.xlabel('Image Size') 
plt.ylabel('Accuracy (%)') 
plt.title('Accuracy Vs Image Size') 
plt.show() 

