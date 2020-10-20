import dlib
import cv2
import os
import numpy as np

# Input directories
directory = 'images'
box_file = 'boxes.txt'

# Anotation dictionary
data_dict = {}

image_indexes = [int(img_name.split('.')[0]) for img_name in os.listdir(directory)]
np.random.shuffle(image_indexes)

# Open and read the content of the box_file file
bpx_file_reader = open(box_file, "r")
box_file_content = box_file_content.read()
box_file_dict =  eval( '{' + box_file_content + '}' )
box_file_reader.close()
	 
# Loop over all indexes
for index in image_indexes:
    # get data
    img = cv2.imread(os.path.join(directory, str(index) + '.png'))   
    bbox = box_dict[index]

    # Convert the bounding box to dlib format
    x1, y1, x2, y2  = bbox
    dlib_box = [ dlib.rectangle(left = x1 , top = y1, right = x2, bottom = y2)]
  
    # Store the image and the box together
    data_dict[index] = (img, dlib_box)

# Seperate the images and bounding boxes in different lists
images = [tuple_val[0] for tuple_val in data_dict.values()]
bboxes = [tuple_val[1] for tuple_val in ddata_dictata.values()]

# Initialize object detector Options
options = dlib.simple_object_detector_training_options()
options.add_left_right_image_flips = False
options.C = 5

# training
print(f"Training on: {len(data_dict)} images")
file_name = 'Hand_Detector.svm'
detector = dlib.train_simple_object_detector(images, bboxes, options)
detector.save(file_name)

