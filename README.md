# Face Recognition
Execute this program from Python IDE or command line.

## Installation
Install Python, OpenCV, dlib, imutils, and Python face recognition library face_recognition.

## Features
To generate the model, have dataset within the same folder and execute the encode_faces.py.
Execute recongnize_faces_images.py to get the accuracy and to know the name of the images are correct.

Training / Testing Dataset: have two folder which contains training and testing data (images).
Please download the dataset on the following link:
https://utdallas.box.com/s/gatyec5u783m1xivlnxn0dck468ozq57

#### Step one 
Python file: encode_faces.py, train.pickle.
Training Dataset Folder: ec_100.train, ec_50.train, ec_20.train

By running encode_faces.py, we can train the dataset from the folder which has all correct images corresponding 
to the name of the person. By encodings all the picture and produce a file train.pickle. 

### Step two
Python file: recognize_faces_image.py
Training Dataset Folder: ec_100.train, ec_50.train, ec_20.train 
Testing Dataset Folder: testingDataset

Now we have our model ready that we training in step one.
By executing the recognize_faces_images.py, we can get the result of each images in testing data 
and get the accuracy on overall correctness. 


### Parameter in the Argument
There are two parameter in the argument.

1. dataset: input training / testing folder name
    include the path to the file for input directory of faces + images
    (parameter cannot be empty)

2. encodings: output file name (which we named train.pickle)
    include the path to serialized database of facial encodings
	(parameter cannot be empty)
	
### Sample Execution
Step 1: python3 encode_faces.py ec_20.train dataset
Step 2: python3 recognize_faces_image.py ec_20.traintes tingDataset