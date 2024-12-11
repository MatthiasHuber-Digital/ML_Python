import inference
from roboflow import Roboflow
import super_gradients

%cd {HOME}

import roboflow
from roboflow import Roboflow

roboflow.login()

rf = Roboflow()

project = rf.workspace("matthias").project("fa")
dataset = project.version(1).download("yolov5")

#model = inference.get_model("fa-mibl4/1")


# Step 1: Initialize the Roboflow object with your API key
#rf = Roboflow(api_key="2O69OR3dbWuYVVI7KUyI")  # Replace with your Roboflow API key

# Step 2: Access your project (replace with your project name and version number)
#project = rf.workspace("matthias").project("fa-mibl4")
#version = project.version(1)  # Replace with your version number

# Step 3: Download the model in YOLOv5 format
#model = version.download("yolov7pytorch")  # You can also choose 'tf', 'pytorch', 'onnx', etc.
#model = inference.get_model("fa-mibl4/1")


print("got it")