# Chien-Chi Liu, Yi-Yuan Chiang, Yi Ting Chen
# Term Project: facial recognition


import face_recognition
import pickle
import cv2
import sys
import os
import re


def scan_dataset(folder):
    file_list = []
    print("scanning dataset...")
    for (rootDir, dirNames, fileNames) in os.walk(folder):
        for filename in fileNames:
            if re.match(r'.*\.(jpg|jpeg|png)', filename, flags=re.I):
                file_list.append(os.path.join(rootDir, filename))

    return file_list


def encode(dataset_folder, encoding_file):
    image_paths = scan_dataset(dataset_folder)

    # initial trained face encoding array and known names arrays
    train_encodings = []
    train_names = []

    # loop through the path of the images
    for (i, imagePath) in enumerate(image_paths):

        print("image {}/{}: ".format(i + 1, len(image_paths)) + imagePath)
        # get the name of the person from the path of the image
        name = imagePath.split(os.path.sep)[-2]

        # convert the RGB image of the input to dlib RGB ordering
        rgb_image = cv2.imread(imagePath)
        rgb = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)

        # locate face in the image and return (x, y)-coordinates
        boxes = face_recognition.face_locations(rgb)

        # compute face image into 128-dimension encodings
        encodings = face_recognition.face_encodings(rgb, boxes)

        # store encode results with correspond name
        for j in encodings:
            # give encoding j variable and append name to the train face name and encoding
            train_encodings.append(j)
            train_names.append(name)

    print("saving encoded face data as a serialized file...")
    data = {"encodings": train_encodings, "names": train_names}
    f = open(encoding_file, "wb")
    f.write(pickle.dumps(data))
    f.close()
    print('done')


if __name__ == '__main__':
    if len(sys.argv[1:]) != 2:
        print('Please Enter 2 Input Arguments')
        exit(0)

    dataset_folder = sys.argv[1]
    encoding_file = sys.argv[2]
    encode(dataset_folder, encoding_file)
