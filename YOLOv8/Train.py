from ultralytics import YOLO 

def main():

    model = YOLO("yolov8n.yaml")

    # use the config file 
    data_yaml_file = "dataset/data.yaml"

    project = "."
    experiment = "Model-65e"

    batch_size = 32

    # train the model 
    results = model.train(data=data_yaml_file,
                          epochs=65,
                          project=project, 
                          name = experiment , 
                          batch = batch_size , 
                          device = 0,
                          patience = 0,
                          imgsz=640 , 
                          verbose = True ,
                          val=True)

if __name__ == "__main__":
    main()
