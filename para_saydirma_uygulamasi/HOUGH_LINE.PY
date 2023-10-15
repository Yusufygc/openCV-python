import cv2
import numpy as np

# Görüntüyü oku
image = cv2.imread('C:\\Users\\TUF Dash F15\\Desktop\\KODLAR\\IMAGE_PROCESSING\\cvCourse\\para_saydirma_uygulamasi\\para.jpg')

# Görüntüyü gri tonlamaya dönüştür
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Kenarları tespit etmek için Canny kenar dedektörünü kullan
edges = cv2.Canny(gray, threshold1=140, threshold2=200)

# Kenarları bulanıklaştırarak gürültüyü azalt
blurred = cv2.GaussianBlur(edges, (7, 7), 0)

# Daireleri tespit etmek için Hough dönüşümünü kullan
circles = cv2.HoughCircles(
    blurred,
    cv2.HOUGH_GRADIENT,
    dp=1,
    minDist=20,
    param1=220,
    param2=23,
    minRadius=5,
    maxRadius=40
) 

if circles is not None:
    # Bulunan daireleri çiz
    circles = np.uint16(np.around(circles)) # yaklaşık değerleri yuvarla
    for circle in circles[0, :]: # her bir daire için
        center = (circle[0], circle[1]) # merkez noktası
        radius = circle[2] # yarıçapı
        # Her bir parayı çizmek için bir çember çizin
        cv2.circle(image, center, radius, (0, 255, 0), 2)

 # Tespit edilen dairelerin sayısını yazdır
    num_circles = len(circles[0])
    print(f"Tespit edilen paraların sayısı: {num_circles}")

# Sonuçları göster
cv2.imshow('Tespit Edilen Paralar', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

"""
cv2.HoughCircles, OpenCV kütüphanesindeki bir işlevdir ve görüntüde dairesel nesneleri tespit etmek için kullanılır. Genellikle Hough Dönüşümü'nün özel bir varyantı olan Hough Daire Dönüşümü ile dairesel nesneleri algılamak için kullanılır. Hough Daire Dönüşümü, görüntüdeki dairesel şekillerin (örneğin, yuvarlak nesnelerin) merkezlerini ve yarıçaplarını bulma işlemidir.

cv2.HoughCircles işlevinin temel parametreleri şunlardır:

image: Daireleri tespit etmek için kullanılacak gri tonlamalı veya kenar algılanmış görüntü.
method: Daire tespiti için kullanılacak yöntem. cv2.HOUGH_GRADIENT en yaygın kullanılan yöntemdir.
dp: Hough dönüşümü çözünürlüğüdür.
minDist: Algılanan daireler arasındaki minimum mesafe.
param1: Canny kenar dedektörü için birinci eşik.
param2: Hough dönüşümü için ikinci eşik.
minRadius: Algılanabilecek dairelerin minimum yarıçapı.
maxRadius: Algılanabilecek dairelerin maksimum yarıçapı.
cv2.HoughCircles işlevi, algılanan dairelerin merkezlerini, yarıçaplarını ve bazen diğer parametreleri içeren bir dizi döndürür. Bu bilgileri kullanarak tespit edilen daireleri işaretleyebilir veya analiz edebilirsiniz.

Özetle, cv2.HoughCircles, görüntüdeki daireleri tespit etmek ve bu dairelerin özelliklerini elde etmek için kullanılan bir araçtır. Genellikle nesne tanıma ve görüntü işleme uygulamalarında dairesel nesneleri algılamak için kullanılır.
"""