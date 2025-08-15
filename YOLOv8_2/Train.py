from ultralytics import YOLO 

def main():

    model = YOLO("yolov8l.yaml")

    # use the config file 
    data_yaml_file = "D:/OneDrive - Rose-Hulman Institute of Technology/Rose-Hulman/course/CSSE/CSSE463/final project/git/Poker-Card-Recognition/YOLOv8_2/dataset/data.yaml"


    project = "D:/OneDrive - Rose-Hulman Institute of Technology/Rose-Hulman/course/CSSE/CSSE463/final project/git/Poker-Card-Recognition/YOLOv8_2/Models"

    experiment = "Model_35_epoch"

    batch_size = 32

    # train the model 
    results = model.train(data=data_yaml_file,
                          epochs=35,
                          project=project, 
                          name = experiment , 
                          batch = batch_size , 
                          device = "cpu",
                          patience = 0,
                          imgsz=640 , 
                          verbose = True ,
                          val=True)

if __name__ == "__main__":
    main()
