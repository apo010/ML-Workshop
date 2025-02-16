import cv2

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Kamera Açılmadı!")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Görüntü alınamadı!")
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_frame, 50,150)

    cv2.rectangle(frame,(100,100),(300,300),(255,0,0),2)

    cv2.putText(frame,"OPENCV!",(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)

    cv2.imshow("Kamera Görüntüsü:", frame)
    cv2.imshow("Gri Kamera", gray_frame)
    cv2.imshow("Kenar Algılama", edges)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


