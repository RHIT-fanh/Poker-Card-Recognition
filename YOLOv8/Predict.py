from ultralytics import YOLO
import cv2 
import os 

imgTest = "./dataset/test/images/004452289_jpg.rf.52bab83a2eae97597af73bfae5f29d2c.jpg"

img = cv2.imread(imgTest)
H , W , _ = img.shape

# predict 
imgpredict = img.copy()
model_path = os.path.join("./","Model3","weights","best.pt")

# load the model 
model = YOLO(model_path)
threshold = 0.5 

results = model(imgpredict)[0]

print(results)

for result in results.boxes.data.tolist():
    x1 , y1 , x2 , y2 , score, class_id = result 

    if score > threshold:
        cv2.rectangle(imgpredict, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 1)
        cv2.putText(imgpredict, results.names[int(class_id)].upper(), (int(x1), int(y1-10)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)

cv2.imwrite("c:/temp/imgpredict.png",imgpredict)
cv2.imshow("imgpredict", imgpredict)

cv2.waitKey(0)

