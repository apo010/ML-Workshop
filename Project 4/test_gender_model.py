import cv2
import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model("gender_model.h5")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(grey, scaleFactor=1.3, minNeighbors=5,minSize=(30,30))

    for(x,y,w,h) in faces:
        face = frame[y:y+h, x:x+w]
        face = cv2.resize(face,(128,128))
        face = np.expand_dims(face,axis=0)
        face = face / 255.0

        prediciton = model.predict(face)[0][0]
        label = "Erkek" if prediciton > 0.5 else "Kadin"

        cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
        cv2.putText(frame, label, (x,y -10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)

    cv2.imshow("Cinsiyet Tahmini:", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

