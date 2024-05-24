#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
print(sys.version)


# In[2]:


# Install necessary packages
get_ipython().system('pip install pillow matplotlib scikit-learn scipy opencv-python-headless')

# Additional packages you might need
get_ipython().system('pip install numpy tornado')


# In[19]:


import cv2
import glob
import os
import pprint
import datetime
import time
import numpy as np
from matplotlib.colors import Normalize
from sklearn import svm
from sklearn.model_selection import GridSearchCV
import pickle
from PIL import Image
from sklearn.preprocessing import StandardScaler

pp = pprint.PrettyPrinter(indent=4)


class SketchData:
    """Support Vector Machine for Sketch recognition"""

    def __init__(self, size, step):

        self.keypoints = self.create_keypoints(size, size, step)

        # histogram orientations: 4 neighbors horizontal * 4 neighbors vertical * 8 directions
        num_entry_per_keypoint = 4 * 4 * 8

        self.descriptor_length = num_entry_per_keypoint * len(self.keypoints)

        self.timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%y-%m-%d_%H:%M:%S')

        self.categories = []

        # Initiate ORB detector
        self.orb = cv2.ORB_create()

    def get_keypoints(self):
        return self.keypoints

    def load_images(self, train_path):
        # read categories by folders
        self.categories = sorted(glob.glob(train_path))
        categories_and_images = []
        num_train_images = 0

        """read all image paths in each category folder"""
        for cat in self.categories:
            category_name = os.path.basename(os.path.normpath(cat))
            images_in_cat = glob.glob(cat + '/*.png')
            num_train_images += len(images_in_cat)
            categories_and_images.append((category_name, images_in_cat))

        return categories_and_images, num_train_images

    def load_images_google(self, train_path):
        categories = sorted(glob.glob(train_path))
        categories_and_images = []
        num_train_images = 0

        pprint.pprint(categories)

        for test_filename in categories:
            images = np.load(test_filename)

            images_formatted = []
            for image in images:
                image_pxl = image.reshape(28, 28)
                image_pxl = np.invert(image_pxl)
                images_formatted.append(image_pxl)

            num_train_images += len(images_formatted)
            categories_and_images.append((test_filename, images_formatted))

        return categories_and_images, num_train_images

    def create_keypoints(self, w, h, keypoint_size):
        """
        creating keypoints on grid for later image segmentation
        :param w: width of grid
        :param h: height of grid
        :param keypoint_size: keypoint size
        :return: array of kreypoints
        """
        keypoints_list = []

        for x in range(0, w, keypoint_size):
            for y in range(0, h, keypoint_size):
                keypoints_list.append(cv2.KeyPoint(x, y, keypoint_size))

        return keypoints_list

    def create_orb_descriptors_for_image(self, image):
        # compute descriptors for each keypoint using ORB
        _, des = self.orb.compute(image, self.keypoints)
        return des

    def get_training_data(self, google, path, sift=True):
        if google:
            categories_and_images, num_train_images = self.load_images_google(path)
        else:
            categories_and_images, num_train_images = self.load_images(path)

        print("loaded %d images" % num_train_images)

        if sift:
            deslen = self.descriptor_length
        else:
            deslen = 28 * 28

        # create y_train vector containing the labels as integers
        y_train = np.zeros(num_train_images, dtype=int)

        # x_train matrix containing descriptors as vectors
        x_train = np.zeros((num_train_images, deslen))

        index_img = 0
        index_cat = 0

        for (cat, image_filenames) in categories_and_images:
            for image in image_filenames:
                if google:
                    image_read = image
                else:
                    image_read = cv2.imread(image, 0)

                if sift:
                    des = self.create_orb_descriptors_for_image(image_read)
                else:
                    des = image_read

                # each descriptor (set of features) needs to be flattened into one vector
                x_train[index_img] = des.flatten()
                y_train[index_img] = index_cat
                index_img = index_img + 1
            index_cat = index_cat + 1

        return x_train, y_train

    def get_name_for_category(self, category):
        return self.categories[category]


class SketchSvm:
    """Support Vector Machine for Sketch recognition"""


    def __init__(self, size, step):

        self.sketchdata = SketchData(size, step)

        self.keypoints = self.sketchdata.get_keypoints()
        print("created " + str(len(self.keypoints)) + " keypoints")

        # histogram orientations: 4 neighbors horizontal * 4 neighbors vertical * 8 directions
        num_entry_per_keypoint = 4 * 4 * 8

        self.descriptor_length = num_entry_per_keypoint * len(self.keypoints)

        self.timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%y-%m-%d_%H:%M:%S')

        self.scaler = StandardScaler()

    def fit_scaler(self, quickdraw, trainpath):
        x_train, y_train = self.sketchdata.get_training_data(quickdraw, trainpath, sift=True)
        self.scaler.fit(x_train)

    def load_model(self, model_file):
        if not model_file:
            print("model name has to be provided")
            exit(0)

        if not os.path.isfile(model_file):
            print("model not found")
            exit(0)

        return pickle.load(open(model_file, 'rb'))

    def draw_heatmap(self, google, model, params, kernel):
        if google:
            path = "results-quickdraw/"
        else:
            path = "results/"
        fig = plot.grid_search(model.grid_scores_, change=params, normalize=MidpointNormalize(midpoint=0.7))
        fig.get_figure().savefig(path + self.timestamp + "_" + kernel + ".pdf")

    def save_model(self, google, model):
        if google:
            save_location = 'models-quickdraw/'
        else:
            save_location = 'models/'
        pickle.dump(model, open(save_location + self.timestamp + '.sav', 'wb'))

    def train(self, quickdraw, path, c_range, gamma_range, kernel, save=True):
        x_train, y_train = self.sketchdata.get_training_data(quickdraw, path, sift=False)
        x_train = self.scaler.fit_transform(x_train)

        parameters = {'C': c_range, "gamma": gamma_range}

        clf = GridSearchCV(svm.SVC(kernel=kernel), parameters)
        clf.fit(x_train, y_train)

        if save:
            self.save_model(quickdraw, clf)

        self.draw_heatmap(quickdraw, clf, ('C', 'gamma'), kernel)

        return clf

    def test_image(self, image, model):
        des = self.sketchdata.create_orb_descriptors_for_image(image)

        test_descriptor = np.zeros((1, self.descriptor_length))
        test_descriptor[0] = des.flatten()

        test_descriptor = self.scaler.transform(test_descriptor)

        label = model.best_estimator_.predict(test_descriptor)

        return label[0]

    def test_model(self, model, testpath):
        test_images = glob.glob(testpath)

        for test_filename in test_images:
            image_read = cv2.imread(test_filename, 0)
            label = self.test_image(image_read, model)
            print("label of " + test_filename + " is predicted as \"" + self.sketchdata.get_name_for_category(
                label) + "\"")

    def test_google(self, model, googletestpath):
        test_images = glob.glob(googletestpath)

        for test_filename in test_images:
            images = np.load(test_filename)

            for image in images:
                image_pxl = image.reshape(28, 28)
                image_pxl = np.invert(image_pxl)
                im = Image.fromarray(image_pxl)
                im.show()
                label = self.test_image(image_pxl, model)
                print("label of " + test_filename + " is predicted as \"" + self.sketchdata.get_name_for_category(
                    label) + "\"")

import time

start_time = time.time()

svm = SketchSvm(28, 7)

quickdraw_path = '"C:/Users/Surayia Rahman/Downloads/quickdraw-data'
c_range = [1, 10, 100]
gamma_range = [.0001, .001]

model = svm.train(True, quickdraw_path, c_range, gamma_range, kernel="rbf")
print("The best parameters are %s with a score of %0.2f" % (model.best_params_, model.best_score_))

quickdraw_path_test = '"C:/Users/Surayia Rahman/Downloads/quickdraw-data/quickdraw-test'
svm.test_google(model, quickdraw_path_test)

print("--- %0.2f minutes ---" % ((time.time() - start_time) / 60))

