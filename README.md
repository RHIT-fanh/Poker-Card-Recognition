# Poker-Card-Recognition
Poker Card Recognition

install dependency `pip install -r requirements.txt`
run main.py on localhost with `python -m streamlit run main.py`


### Folder Hu:
This folder contains some trial on using Hu invariant moment to do template matching for poker card recognition:

#### get_roi_mask:
This function uses the area property of connected components to get numbers and suits of the card
#### computeHuMoments:
This function compute the seven Hu invariant moments
#### noneML:
This code is the main code where input an image, extract Hu moments, get distance with all the templates and output the recognized number or suit
#### extract_template_hu:
This code extract the Hu moments of the template
#### flipping:
This code flips the foreground to white and background to 0
#### size_test:
This code proves Hu moments remains unchanged when resizing
#### spin_test:
This code proves Hu moments remains unchanged when rotating
#### shear_test:
This code proves Hu moments remains unchanged when shearing
#### strech_test:
This code proves Hu moments changes when streching


### Folder YOLOv8_2:
This folder is training the model with another dataset:  https://huggingface.co/datasets/JackFurby/playing-cards
A 35 epochs model is trained for this dataset, this is for YOLO whole card detection with machine generated, no blocking images 

### Folder YOLOv8_3:
This folder is training the model with another dataset:  https://universe.roboflow.com/roboflow-jvuqo/poker-cards-fmjio
35/50/65 epochs models are trained for this dataset, this is for YOLO whole card detection with real world, blocking images
