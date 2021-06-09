## 3D Bounding Boxes

The 3D bounding box dataset contains 12,497 frames. Each frame has the
following items:

- RGB image
- 3D point cloud
- annotated semantic segmentation label
- annotated 3D bounding box label

All frames are grouped in 18 different scenes with each scene contained
in its corresponding folder. Scene folder names are in the
'YYYYMMDD_hhmmss' format. They represents the date and time of the
recording. Each scene is further divided into four folders:

- 'camera': input images and json info files
- 'lidar': input 3D point clouds
- 'label': annotated label images
- 'label3D': annotated 3D bounding boxes 

Each of these folders are further divided depending on the camera from
which the data was recorded. There are six cameras available in the
vehicle, therefore, the following are the camera folders:

- 'cam_front_center'
- 'cam_front_left'
- 'cam_front_right'
- 'cam_side_left'
- 'cam_side_right'
- 'cam_rear_center'

Lastly, each of these folders contains the corresponding item for each
frame.

These are the filename formats for the item of a single frame:

input RGB image  : YYMMDDDDhhmmss_camera_[frontcenter|frontleft|frontright|sideleft|sideright|rearcenter]_[ID].png
input info       : YYMMDDDDhhmmss_camera_[frontcenter|frontleft|frontright|sideleft|sideright|rearcenter]_[ID].json
3D lidar pcloud  : YYMMDDDDhhmmss_lidar_[frontcenter|frontleft|frontright|sideleft|sideright|rearcenter]_[ID].npz
label image      : YYMMDDDDhhmmss_label_[frontcenter|frontleft|frontright|sideleft|sideright|rearcenter]_[ID].png
3D Bounding Box  : YYMMDDDDhhmmss_label3D_[frontcenter|frontleft|frontright|sideleft|sideright|rearcenter]_[ID].json

For example, a frame with ID 1617 from a scene recorded on 2018-08-07
14:50:28 from the front center camera would be have the following items
in these locations:

input RGB image  : 20180807_145028/camera/cam_front_center/20180807145028_camera_frontcenter_000001617.png
input info       : 20180807_145028/camera/cam_front_center/20180807145028_camera_frontcenter_000001617.json
3D lidar pcloud  : 20180807_145028/lidar/cam_front_center/20180807145028_lidar_frontcenter_000001617.npz
label image      : 20180807_145028/label/cam_front_center/20180807145028_label_frontcenter_000001617.png
3D Bounding Box  : 20180807_145028/label3D/cam_front_center/20180807145028_label3D_frontcenter_000001617.json

For further explanations regarding the format of each of these items,
please refer to the tutorial in our dataset web page.
