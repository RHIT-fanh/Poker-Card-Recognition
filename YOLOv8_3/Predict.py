from ultralytics import YOLO
import cv2 
import os 

# imgTest = r"D:\OneDrive - Rose-Hulman Institute of Technology\Rose-Hulman\course\CSSE\CSSE463\final project\git_latest\YOLOv8_3\poker cards.v4i.yolov8\test\images\IMG_20220316_171936_jpg.rf.b6e31b1cc6b5e14dc66462becfa4a63d.jpg"
# imgTest = r"D:\OneDrive - Rose-Hulman Institute of Technology\Rose-Hulman\course\CSSE\CSSE463\final project\git_latest\YOLOv8_2\dataset\test\images\7346_three_card_poker_class_level.png"
# imgTest = r"D:\OneDrive - Rose-Hulman Institute of Technology\Rose-Hulman\course\CSSE\CSSE463\final project\git_latest\YOLOv8\dataset\test\images\596131312_jpg.rf.3f88c1789ff78e4bdeccc7a1d5baf579.jpg"
# imgTest = r"D:\OneDrive - Rose-Hulman Institute of Technology\Rose-Hulman\course\CSSE\CSSE463\final project\git_latest\Hu\dataset\Images\Images\2H11.jpg"
imgTest = r"C:\Users\PC\Desktop\临时文件\testing.jpg"


img = cv2.imread(imgTest)
H , W , _ = img.shape

# predict 
imgpredict = img.copy()
# imgpredict = cv2.resize(imgpredict, (640, 480))
model_path = r"D:\OneDrive - Rose-Hulman Institute of Technology\Rose-Hulman\course\CSSE\CSSE463\final project\git_latest\YOLOv8_3\Models\Model_65_epoch\weights\best.pt"
# model_path = r"D:\OneDrive - Rose-Hulman Institute of Technology\Rose-Hulman\course\CSSE\CSSE463\final project\git_latest\YOLOv8\Model-65e\weights\best.pt"
# model_path = r"D:\OneDrive - Rose-Hulman Institute of Technology\Rose-Hulman\course\CSSE\CSSE463\final project\git_latest\YOLOv8_2\Models\Model_35_epoch\weights\best.pt"


# load the model 
model = YOLO(model_path)
threshold = 0.1 

results = model(imgpredict)[0]

print(results)

for result in results.boxes.data.tolist():
    x1 , y1 , x2 , y2 , score, class_id = result 

    if score > threshold:
        # 画框
        cv2.rectangle(imgpredict, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 1)

        # 计算框中心点
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        label = results.names[int(class_id)].upper()

        # 获取文字尺寸 (w, h)
        (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)

        # 让文字居中：x = 中心 - 宽度/2, y = 中心 + 高度/2
        text_x = cx - text_w // 2
        text_y = cy + text_h // 2

        # 写文字
        cv2.putText(imgpredict, label, (text_x, text_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2, cv2.LINE_AA)


cv2.imwrite("c:/temp/imgpredict.png",imgpredict)
cv2.imshow("imgpredict", imgpredict)

cv2.waitKey(0)

