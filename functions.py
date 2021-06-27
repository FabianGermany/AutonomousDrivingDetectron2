#Functions needed to handle the Audi A2D2 dataset, according to https://www.a2d2.audi/a2d2/en/tutorial.html
#There are more functions, but I don't need all of them since some of them focus on 3D bounding boxes
#--------------------------------------------------------

#the imports are not needed here because they are already mentioned in the Colab file
import json, pprint
import numpy as np
import numpy.linalg as la
import cv2
#import open3d as o3

#this is used for the function axis_angle_to_rotation_mat
def skew_sym_matrix(u):
    return np.array([[    0, -u[2],  u[1]], 
                     [ u[2],     0, -u[0]], 
                     [-u[1],  u[0],    0]])


#this is used for the function read_bounding_boxes
def axis_angle_to_rotation_mat(axis, angle):
    return np.cos(angle) * np.eye(3) + \
        np.sin(angle) * skew_sym_matrix(axis) + \
        (1 - np.cos(angle)) * np.outer(axis, axis)


def read_bounding_boxes(file_name_bboxes, mute = True):
    # open the file
    with open (file_name_bboxes, 'r') as f:
        bboxes = json.load(f)
        
    boxes = [] # a list for containing bounding boxes  
    if(not mute): print(bboxes.keys())
    
    for bbox in bboxes.keys():
        bbox_read = {} # a dictionary for a given bounding box
        bbox_read['class'] = bboxes[bbox]['class']
        bbox_read['truncation']= bboxes[bbox]['truncation']
        bbox_read['occlusion']= bboxes[bbox]['occlusion']
        bbox_read['alpha']= bboxes[bbox]['alpha']
        bbox_read['top'] = bboxes[bbox]['2d_bbox'][0]
        bbox_read['left'] = bboxes[bbox]['2d_bbox'][1]
        bbox_read['bottom'] = bboxes[bbox]['2d_bbox'][2]
        bbox_read['right']= bboxes[bbox]['2d_bbox'][3]
        bbox_read['center'] =  np.array(bboxes[bbox]['center'])
        bbox_read['size'] =  np.array(bboxes[bbox]['size'])
        angle = bboxes[bbox]['rot_angle']
        axis = np.array(bboxes[bbox]['axis'])
        bbox_read['rotation'] = axis_angle_to_rotation_mat(axis, angle) 
        boxes.append(bbox_read)

    return boxes 

#extract file name from image file name (I don't use this cause I defined my own procedure for that)
def extract_bboxes_file_name_from_image_file_name(file_name_image):
    file_name_bboxes = file_name_image.split('/')
    file_name_bboxes = file_name_bboxes[-1].split('.')[0]
    file_name_bboxes = file_name_bboxes.split('_')
    file_name_bboxes = file_name_bboxes[0] + '_' + \
                  'label3D_' + \
                  file_name_bboxes[2] + '_' + \
                  file_name_bboxes[3] + '.json'
    
    return file_name_bboxes



#functions for frequent usage
#--------------------------------------------------------

#resize image for display into notebook
def resize_img (scale_percent, image):
  width = int(image.shape[1] * scale_percent / 100)
  height = int(image.shape[0] * scale_percent / 100)
  dim = (width, height)
  image_resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
  return image_resized