# Object Detection with Detectron2 finetuned on Audi A2D2

## Table of Contents

1. [About](#about)
2. [How to run](#how-to-run)
    1. [Set up the Jupyter notebook](#set-up-the-jupyter-notebook)
    2. [Prepare dataset](#prepare-the-dataset)
    3. [Choose (Hyper-)Parameters](#before-run)
    4. [Run](#run)
3. [FAQ](#problems)


 <a name="about"></a>
## About

This repository is about an object detection system using Detectron2 algorithm and Audi A2D2 dataset for autonomous vehicles.

[Screenshot](output_data/exemplary_images/example_output_bounding_box_Faster_R_CNN_trained.jpg)


![Video Bounding Boxes Trained](output_data/scene_2/exemplary_scene_rural_2_muted_output_bounding_box_Faster_R_CNN_trained_compressed.gif)


For this project, I use [Detectron2](https://github.com/facebookresearch/detectron2) from Facebook  to train and test data
from [Audi](https://www.a2d2.audi/a2d2/en.html). 
Complete citatations are mentioned in my thesis that is not published here in GitHub. Summing up, with this software we will train a Detectron2 model that you can choose by yourself (e.g. Faster R-CNN) on the 2D-Bounding-Box information from Audi A2D2. This software could also be extended by further features since Audi A2D2 also features 3D bounding boxes, semantic segmentation etc. In the video below, we can see that with Detectron2 we're also able to handle semantic segmentation, panoptic segmentation etc. But let's first start with 2D bounding boxes as shown in the video above.

![Video Panoptic Segmentation Pretrained](output_data/scene_1/exemplary_scene_rural_1_muted_output_panoptic_segmentation_Panoptic_FPN_pretrained_compressed.gif)



<a name="how-to-run"></a>
## How to run 

This is about how to run this software.


Due to the required GPU capacities, this project is embedded into a Google-Colab notebook so that the training and testing of the model can be outsourced to the Google Server. The notebook is based on the [example notebook](https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5) from Facebook's Detectron2. Documentation about the Audi A2D2 dataset can be found [here](https://www.a2d2.audi/a2d2/en/tutorial.html).
Another reason for outsourcing the script to Google Colab is that Detectron2 hardly works on Windows systems, so otherwise is the need for Windows users to create a Linux virtual machine with specific package dependencies.

<a name="set-up-the-jupyter-notebook"></a>
### Set up the Jupyter notebook

Please use the [GitHub link](https://colab.research.google.com/github/FabianGermany/AutonomousDrivingDetectron2/blob/main/Detectron2_Personal_Notebook_GoogleDrive_Instance.ipynb) to the file of the notebook inside GitHub. Then save this notebook to your Google Drive and keep the original name of this file. Then please make changes on the file inside your Google Drive. Once changes are made to this notebook, commit and push the changes into the GitHub repo using Colab's function "File --> Save a copy into your GitHub". This way worked better for me than the other way (every time importing the current notebook from GitHub to Google Drive) because of several reasons. With the way I proposed, you can also make sure that you can use files stored in your Google Drive for your notebook.

Consequently, the heart of this project is the Google-Colab notebook file. It can be run locally, on a Google-Colab server or after some preparations on a high performance server such as Amazon AWS S3 SageMaker. In my case, I'll use it on a Google Server because Google Colab is for free to some extent.

<a name="prepare-the-dataset"></a>
### Prepare the dataset

You can download the 3D Bounding Boxes dataset from [Audi A2D2 website](https://www.a2d2.audi/a2d2/en/download.html).
Make sure, that your dataset will be stored on your Google Drive inside the folder structure `content/gdrive/My Drive/Dev/ColabDetectron2Project`.
There you have two zip-files called `dataset_1_train.zip` and `dataset_2_train.zip` that include the desired sub-datasets in the format given by Audi. After unzipping, an exemplary .json file with the bounding box information has the path
`content/gdrive/My Drive/Dev/ColabDetectron2Project\dataset_1_train\20181107_132730\label3D\cam_front_center\20181107132730_label3D_frontcenter_000000135.json`. If you use different sub-datasets than me, make sure to replace the path components like '20181107_132730' by your own paths. In my case I only use three folders of the dataset because my Google Drive storage is not enough to handle to whole dataset. For better results and for avoiding overfitting it's highly recommended to use more data.

The software will take care of converting the dataset to Detectron2 format by itself, so dont't worry about that.

<a name="before-run"></a>
### Choose (Hyper-)Parameters

Before running, make sure to choose the desired parameters. This includes the sub-datasets paths as mentioned above, but also the choice whether you want to install Detectron2 permanently on your Google Drive or not using `local_install` variable.  Also we have `dataset_json_available` . Once the script has run at least once, we can set it to `True` in order to load the stored .json file so we don't need to re-parse every time. Furthermore we have `load_existing_trained_model`. We can set this to `True` so that we can load our trained model. This is useful if we only want to evaluate something and we already ran this script before, but we don't want to re-train our pre-trained model everytime we launch Colab to save time and ressources. If you want to skip the evaluation and/or the metrics calculation part, you can use the bool values `run_evaluation` and `calculate_metrics` for that.

Also make sure to choose the desired model in section2. There you can pick `model_path = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"` for using a Faster R-CNN object detection model, for example. If you want to choose a completely different one, please make sure that the configurations name still match with the given ones. For example `MODEL.ROI_HEADS.NUM_CLASSES` doesn't work for RetinaNet. It must be called `MODEL.RETINANET.NUM_CLASSES` instead. See more [here](https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets). However, I prepared this script so that it will support Faster R-CNN and RetinaNet so far. This means for other models you may make a few adaptions for the `cfg` data, but also for other stuff like replacing `draw_instance_predictions(outputs["instances"].to("cpu"))` by `draw_panoptic_seg_predictions(outputs["panoptic_seg"].to("cpu"))` for visualizing images if you want to use panoptic segmentation instead of bounding boxes or instance segmentation.


<a name="run"></a>
### Run

Starting from the top of the notebook, just click on the desired scrips. When running the steps for installating some dependencies, you may ask to re-connect to the runtime. In case of that please follow the instructions.


<a name="problems"></a>
## FAQ

If you have some problems executing some commands, it sometimes helps if you delete the output folder and re-run the script:

* If you receive `AssertionError: Results do not correspond to current coco set` at `print(inference_on_dataset(trainer.model, val_loader, evaluator))` then it's actually a bit tricky. According to [this GitHub issue](https://github.com/facebookresearch/detectron2/issues/1631) a way to fix this is to delete all the files in the output folder (the model, the data in .json format etc.). Then you need to reset the Colab runtime so that everything including the installation of Detectron2, the connection to Google Drive etc. will start from the beginning again. After this simply re-run the whole notebook. Maybe it's sufficient to delete one single file or do less steps, but I couldn't find out yet.

* If the GPU is not available or the storage is not sufficient, you may change to another environment such as Amazon AWS or use your own GPU provided you have one. Another solution is to pay for Google Colab Pro what I did.

* If Detectron2 doesn't work, please check whether the right CUDA and torch versions are installed. Detectron2 currently only works for Torch 1.8.0. The problem is that both the libraries versions that are installed on default on Colab and the required packages versions for Detectron2 may change in the course of the time. To make sure please have a look on the [GitHub of Detectron2](https://github.com/facebookresearch/detectron2) and it's [tutorial notebook](https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5)  to check whether there were changes.


<!-- reduce size of gif with tools like https://www.ps2pdf.com/compress-gif -->