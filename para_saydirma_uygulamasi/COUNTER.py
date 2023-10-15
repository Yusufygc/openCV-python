import cv2
import numpy as np

# load image
img = cv2.imread("C:\\Users\\TUF Dash F15\\Desktop\\KODLAR\\IMAGE_PROCESSING\\cvCourse\\para_saydirma_uygulamasi\\para.jpg")

# Görüntüyü gri tonlamaya dönüştür
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Kenarları tespit etmek için Canny kenar dedektörünü kullan
edges = cv2.Canny(gray, threshold1=230, threshold2=130)

# Kenarları bulanıklaştırarak gürültüyü azalt
blurred = cv2.GaussianBlur(edges, (17, 17), 0)

# find the contours
(cnts, _) = cv2.findContours(edges, cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE) # OUTLİNE I EDGES YAPTIM

cv2.drawContours(img, cnts, -1, (0, 255, 0), 2)

print("Tespit edilen paraların sayısı:" , len(cnts)) 

cv2.imshow("Result", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

"""
cv2.findContours, OpenCV (Open Source Computer Vision Library) kütüphanesinde bulunan bir işlevdir ve görüntülerdeki nesnelerin konturlarını (sınırlarını) bulmak için kullanılır. Konturlar, nesnelerin dış hatlarını tanımlayan, nesnelerin kenarlarını oluşturan bir dizi noktayı içerir. Bu işlev, nesne tanıma, görüntü analizi, nesne tespiti ve benzeri birçok uygulamada yaygın olarak kullanılır.

image: Konturları bulmak için kullanılan görüntü.
mode: Kontur tarama modu. Örneğin, cv2.RETR_EXTERNAL, cv2.RETR_LIST, vb. gibi farklı modlar kullanılabilir. Bu mod, kontur yapısını nasıl oluşturacağınızı belirler.
method: Yaklaşım yöntemi. Örneğin, cv2.CHAIN_APPROX_SIMPLE, cv2.CHAIN_APPROX_NONE, vb. gibi farklı yöntemler kullanılabilir. Bu yöntem, kontur verilerini nasıl depolayacağınızı belirler.
cv2.findContours işlevi, konturları ve gerektiğinde konturların hiyerarşisini içeren bir dizi döndürür. Bu dizi, her bir konturun noktalarının koordinatlarını içerir ve her kontur bir liste olarak temsil edilir.

Konturların tespit edilmesi ve analizi, nesne tanıma, nesne sayma, nesnenin boyutlarını ölçme ve benzeri görsel işleme görevlerinde önemlidir. Konturlar, nesnelerin tanınması ve ayrılması için kullanışlıdır ve ayrıca nesne hareketlerini izlemek gibi uygulamalarda da kullanılabilir.
"""