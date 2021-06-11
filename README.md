# Object detection system using Detectron2 algorithm and Audi A2D2 dataset
Object detection system that can be used for autonomous vehicles


![Screenshot](preview_files/example_output.jpg)

For this project, I use Detectron2 from Facebook (https://github.com/facebookresearch/detectron2) to train and test data
from Audi (https://www.a2d2.audi/a2d2/en.html). For further citations have a look on my thesis.

Due to the required GPU capacities, this project is embedded into a Google-Colab notebook so that the training and testing of the model can be outsourced to a Amazon AWS Server. The notebook is based on the example notebook from Facebook's Detectron2 (https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5). Documentation about the Audi A2D2 dataset can be found here: https://www.a2d2.audi/a2d2/en/tutorial.html.
Another reason for outsourcing the script to Colab is that Detectron2 hardly works on Windows systems, so there is the need to create a Linux virtual machine with specific package dependencies.

Consequently, the heart of this project is the Google-Colab notebook file. It can be run locally, on a Google-Colab server or better on a high performance server such as Amazon AWS S3 SageMaker.