# %%
import tensorflow as tf
import cv2
import numpy as np

# %%
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

# %%
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# %%
def detect_objects(image_path):
    img = cv2.imread("./yolov3/frames/" + image_path)
    height, width = img.shape[:2]
    
    # Підготовка зображення для YOLO
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    
    class_ids = []
    confidences = []
    boxes = []
    
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = 'o'#str(classes[class_ids[i]])
            confidence = confidences[i]
            color = 0#colors[class_ids[i]]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label + " " + str(round(confidence, 2)), (x, y + 30), cv2.FONT_HERSHEY_PLAIN, 1, color, 2)
    
    print("./yolov3/results/" + image_path)
    cv2.imwrite("./yolov3/results/" + image_path, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# %%
vidcap = cv2.VideoCapture('./yolo-sample.mp4')
success, image = vidcap.read()
count = 0
while success:
  frame = "frame%d.jpg" % count
  cv2.imwrite("./yolov3/frames/" + frame, image)     # save frame as JPEG file      
  success, image = vidcap.read()
  print('Read a new frame: ', success)
  detect_objects(frame)
  count += 1

# %%
