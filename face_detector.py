import cv2 
import tensorflow as tf

prototxtPath = r"faceDetector\deploy.prototxt.txt"
weightsPath = r"faceDetector\res10_300x300_ssd_iter_140000.caffemodel"

face_net = cv2.dnn.readNet(prototxtPath, weightsPath)
mask_detection = tf.keras.models.load_model("facemask_detector.model")