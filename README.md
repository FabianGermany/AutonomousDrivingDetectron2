# Object Detection with Detectron2 finetuned on Audi A2D2

<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about">About</a>
      <ul>
        <li><a href="#set-up-the-jupyter-notebook">Set up the Jupyter notebook</a></li>
        <li><a href="#prepare-the-dataset">Prepare dataset</a></li>
        <li><a href="#run">Run</a></li>
      </ul>
    </li>
    <li>
      <a href="#how-to-run">How to run</a>
    </li>
  </ol>
</details>



## About

This repository is about an object detection system using Detectron2 algorithm and Audi A2D2 dataset and can be used for autonomous vehicles.

![Screenshot](output_data/example_output_object_detection_pretrained.jpg)


For this project, I use [Detectron2](https://github.com/facebookresearch/detectron2) from Facebook  to train and test data
from [Audi](https://www.a2d2.audi/a2d2/en.html). 
Complete citatations are mentioned in my thesis that is not published here.


![Video](output_data/exemplary_scene_rural_1_muted_output_panoptic_segmentation_pretrained.gif)

## How to run

This is about how to run this software.


Due to the required GPU capacities, this project is embedded into a Google-Colab notebook so that the training and testing of the model can be outsourced to the Google Server. The notebook is based on the [example notebook](https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5) from Facebook's Detectron2. Documentation about the Audi A2D2 dataset can be found [here](https://www.a2d2.audi/a2d2/en/tutorial.html).
Another reason for outsourcing the script to Google Colab is that Detectron2 hardly works on Windows systems, so otherwise is the need for Windows users to create a Linux virtual machine with specific package dependencies.

### Set up the Jupyter notebook

Please us the [GitHub link](https://colab.research.google.com/github/FabianGermany/AutonomousDrivingDetectron2/blob/main/Detectron2_Personal_Notebook_GoogleDrive_Instance.ipynb) to the file of the notebook inside GitHub. Then save this notebook to your Google Drive and keep the original name of this file. Once changes are made to this notebook, commit and push the changes into the GitHub repo using Colab's function "File --> Save a copy into your GitHub".

Consequently, the heart of this project is the Google-Colab notebook file. It can be run locally, on a Google-Colab server or after some preparations on a high performance server such as Amazon AWS S3 SageMaker. In my case, I'll use it on a Google Server because Google Colab is for free to some extent.


### Prepare the dataset

You can download the 3D Bounding Boxes dataset from [Audi A2D2 website](https://www.a2d2.audi/a2d2/en/download.html).
Make sure, that your dataset will be stored on your Google Drive inside the folder structure `content/gdrive/My Drive/Dev/ColabDetectron2Project`.
There you have two zip-files called `dataset_1_train.zip` and `dataset_2_train.zip` that include the desired sub-datasets in the format given by Audi. After unzipping, an exemplary .json file with the bounding box information has the path
`content/gdrive/My Drive/Dev/ColabDetectron2Project\dataset_1_train\20181107_132730\label3D\cam_front_center\20181107132730_label3D_frontcenter_000000135.json`. If you use different sub-datasets than me, make sure to replace the path components like '20181107_132730' by your own paths.

The software will take care of converting the dataset to Detectron2 format by itself, so dont't worry about that.

### Run

Starting from the top of the notebook, just click on the desired scrips. When running the steps for installating some dependencies, you may ask to re-connect to the runtime. In case of that please follow the instructions.