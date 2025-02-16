import cv2

image = cv2.imread("images/sample.jpg")

if image is None:
    print("Resim belirtilen yolda bulunamadı.")
    exit()

cv2.imshow("Orijinal Resim:", image)

cv2.waitKey(0)

cv2.destroyAllWindows()

print(f"Boyutlar: {image.shape}")
print(f"Piksel Sayısı: {image.size}")
print(f"Veri Tipi: {image.dtype}")

roi = image[50:200, 100:200]
cv2.imshow("Kesilen Bölge", roi)

cv2.waitKey(0)

cv2.destroyAllWindows()

resized = cv2.resize(image,(400,300))
cv2.imshow("Yeniden Boyutlandırma:", resized)

cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.line(image,(50,50),(300,50),(0,255,0),2)
cv2.rectangle(image,(100,100),(300,300),(255,0,0),2)
cv2.circle(image,(200,200),50,(0,0,255),3)
cv2.putText(image, "OPENCV",(50,400),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)

cv2.imshow("Cizim Hali:", image)
cv2.waitKey(0)
cv2.destroyAllWindows()