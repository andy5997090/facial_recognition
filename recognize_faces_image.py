# Chien-Chi Liu, Yi-Yuan Chiang, Yi Ting Chen
# Term Project: facial recognition

import face_recognition
import pickle
import sys
import cv2
import os
import re
import time


# find image files in selected folder
def image_files_in_folder(folder):
    file_list = []
    for f in os.listdir(folder):
        if not f.startswith('.') and re.match(r'.*\.(jpg|jpeg|png)', f, flags=re.I):
            file_list.append(os.path.join(folder, f))

    return file_list


def face_rec(train_encodings, image_in):
    # load the known faces and embeddings
    data = pickle.loads(open(train_encodings, "rb").read())

    # load the input image and convert it from BGR to RGB
    image = cv2.imread(image_in)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # find where the face is located in a image and return the rectangle area
    boxes = face_recognition.face_locations(rgb)
    # encode the face in the rectangle area
    test_encodings = face_recognition.face_encodings(rgb, boxes)

    names = []
    if not test_encodings:
        print('Face not detected' + ', No Face')
        return 'No face'

    for test_encoding in test_encodings:
        # attempt to match each face in the input image to our known encodings
        matches = face_recognition.compare_faces(data["encodings"], test_encoding)
        name = "Unknown"

        # check to see if we have found a match
        if True in matches:
            # find the indexes of all matched faces then initialize a
            # dictionary to count the total number of times each face
            # was matched
            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}

            # loop over the matched indexes and maintain a count for
            # each recognized face
            for i in matchedIdxs:
                name = data["names"][i]
                counts[name] = counts.get(name, 0) + 1

            # determine the recognized face with the largest number of votes
            name = max(counts, key=counts.get)
            certainty = counts[name] / len(matchedIdxs)

        # update the list of names
        names.append(name)

        # compare the name to the file to see if it match
        if name.lower() in image_in.lower():
            print(name + ', Hit, ' + 'certainty: {0:.2f}%'.format(certainty * 100))
            return 'Hit'
        elif name == 'Unknown':
            print(name + ', Unknown')
            return 'Unknown'
        else:
            print(name + ', Miss')
            return 'Miss'


if __name__ == '__main__':
    if len(sys.argv[1:]) != 2:
        print('Please Enter 2 Input Arguments')
        exit(0)

    encodings = sys.argv[1]
    image_folder = sys.argv[2]

    image_paths = image_files_in_folder(image_folder)
    total_num = int(len(image_paths))
    hit = 0
    unknown = 0
    no_face = 0
    star_time = time.time()
    for (i, img) in enumerate(image_paths):
        print("processing {}/{}: ".format(i + 1, len(image_paths)), end="")
        print(img.split(os.path.sep)[-1], end=", ")
        result = face_rec(encodings, img)
        if result == 'Hit':
            hit += 1
        elif result == 'Unknown':
            unknown += 1
        elif result == 'No face':
            no_face += 1
        else:
            continue

    end_time = time.time()
    print('Accuracy: {}/{}, '.format(hit, total_num) + '{0:.2f}%'.format(hit / total_num * 100))
    print('unknown: {}'.format(unknown) + ', no face: {}'.format(no_face))
    print('time elapses: {0:.2f} seconds'.format(end_time - star_time))
