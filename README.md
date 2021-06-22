# About

This repository is about an object detection system using Detectron2 algorithm and Audi A2D2 dataset and can be used for autonomous vehicles.

![Screenshot](output_data/example_output_object_detection_pretrained.jpg)


For this project, I use [Detectron2](https://github.com/facebookresearch/detectron2) from Facebook  to train and test data
from [Audi](https://www.a2d2.audi/a2d2/en.html). Complete citatations are mentioned in my thesis.


![Video](output_data/exemplary_scene_rural_1_muted_output_panoptic_segmentation_pretrained.gif)

# How to run
Use the GitHub link to the [file of the notebook](https://colab.research.google.com/github/FabianGermany/AutonomousDrivingDetectron2/blob/main/Detectron2_Personal_Notebook_GoogleDrive_Instance.ipynb "Jupyter Notebook inside GitHub"). Once changes are made, commit and push the changes into this GitHub repo.

Due to the required GPU capacities, this project is embedded into a Google-Colab notebook so that the training and testing of the model can be outsourced to a server such as Google Server or Amazon AWS Server. The notebook is based on the [example notebook](https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5) from Facebook's Detectron2. Documentation about the Audi A2D2 dataset can be found [here](https://www.a2d2.audi/a2d2/en/tutorial.html).
Another reason for outsourcing the script to Colab is that Detectron2 hardly works on Windows systems, so there is the need to create a Linux virtual machine with specific package dependencies.

Consequently, the heart of this project is the Google-Colab notebook file. It can be run locally, on a Google-Colab server or on a high performance server such as Amazon AWS S3 SageMaker. In my case, I'll use it on a Google Server because Amazon has some financial and legal issues.

When running the first steps in the Colab, you may ask to re-connect to the runtime. In case of that please follow the instructions.