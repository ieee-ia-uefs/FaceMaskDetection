import cv2 
import tensorflow as tf
import imutils
import numpy as np
from imutils.video import VideoStream

prototxtPath = r"FaceDetector\deploy.prototxt.txt"
weightsPath = r"FaceDetector\res10_300x300_ssd_iter_140000.caffemodel"

face_net = cv2.dnn.readNet(prototxtPath, weightsPath)
mask_detection = tf.keras.models.load_model("facemask_detector.model")

def detect_and_predict_mask(frame):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),(104.0, 177.0, 123.0))  
    face_net.setInput(blob)
    detections = face_net.forward() 	

    faces = []
    locs = []
    preds = [] 

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = tf.keras.preprocessing.image.img_to_array(face)
            face = tf.keras.applications.mobilenet_v2.preprocess_input(face)

            faces.append(face)
            locs.append((startX, startY, endX, endY))

    if len(faces) > 0:
        faces = np.array(faces, dtype="float32")
        preds = mask_detection.predict(faces, batch_size=32)

    return (locs, preds)

# initialize the video stream
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()

while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=400)

    (locs, preds) = detect_and_predict_mask(frame)

    for (box, pred) in zip(locs, preds):
        (startX, startY, endX, endY) = box
        (mask, withoutMask) = pred

        label = "Com mascara" if mask > withoutMask else "Sem mascara"
        color = (51, 204, 51) if label == "Com mascara" else (26, 26, 255)
        
        label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

        cv2.putText(frame, label, (startX, startY - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cv2.destroyAllWindows()
vs.stop()