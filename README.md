# Poker-Card-Recognition

To launch the Visulization tool
- go to YOLOv8 folder
- install dependency `pip install -r requirements.txt`
- run main.py on localhost with `python -m streamlit run main.py`


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


## 📂 YOLOv8DetectionOnly

This folder contains all scripts and results related to YOLOv8 detection and preprocessing.  

- **infer_pipeline.py** – Runs the full inference pipeline combining detection and classification.  
- **label_obb_script.py** – Helper script for generating rotated bounding box labels.  
- **make_card_only_labels.py** – Prepares labels specifically for card-only detection.  
- **make_cls_dataset.py** – Builds a dataset for the classification step from YOLO outputs.  
- **poker_yolo_detect_and_crop.py** – Detects and crops cards from images using YOLOv8.  
- **train_efficientnet_b0.py** – Training script for EfficientNet-B0 on classification dataset.  

**Folders:**  
- **runs_YOLO/detect** – Stores YOLO detection training logs and outputs.  
- **runs_cls** – Stores classification training logs and outputs.  

---

## 📂 CNNClassification  

This folder contains the CNN-based approach for suit classification.  

- **train_suit_cnn.py** – Script to train a CNN model for suit recognition.  
- **best_suit_resnet18.pth** – Saved weights of the best ResNet18 model.  
- **confusion_matrix_suit.png** – Confusion matrix showing classification performance.  

---
