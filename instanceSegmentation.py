#$env:KMP_DUPLICATE_LIB_OK="TRUE"
from ultralytics import YOLO
import supervision as sv
import cv2
import torch
#anlık segmentasyon yap
# Cihaz kontrolü (GPU varsa kullan)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Kullanılan cihaz: {device}")

# Model yükle
# 'yolov8n-seg.pt' küçük ve hızlı segmentasyon modeli
model = YOLO("yolov8n-seg.pt") #Model, bu komutla görüntüyü analiz ediyor ve tespit ettiği nesnelerin konumlarını (bounding box'lar)
#ve segmentasyon maskelerini içeren bir sonuç (results) döndürüyor.

# Tahmin
# Görsel yolunu değiştir
image_path = "img/cats.png"
results = model(image_path)

# Görselleştir ve pencereye sığdır
# Pencere  boyutu
max_width = 1280
max_height = 720
#projenin görsel çıktısı
for r in results:
    # Bounding box + mask
    #results.plot() metodu: direkt olarak algılanan nesneleri ve segmentasyon maskelerini 
    # çizerek bize görselleştirilmiş bir görüntü sunar.
    annotated_frame = r.plot()
    
    # Orantılı yeniden boyutlandır > belirlenen maksimum değerleri aşmama, orj en boy oranı koruma
    #Farklı ekran boyutlarına uyum sağlaması ve görüntünün ekrana tam sığması için görüntüyü
    # yeniden boyutlandırma
    # görüntünün en boy oranını koruyarak, belirtilen maksimum genişlik ve yüksekliğe göre orantılı
    # bir şekilde yeniden boyutlandırılmasını sağlıyor
    #oranını bozmadan yeniden boyutlandırmak
    #[:2]: Bu Python dilimleme (slicing) işlemi, dizinin ilk iki elemanını, yani yükseklik (h) ve genişliği (w) alır.
    #Renk kanalı sayısı (3) bu aşamada göz ardı edilir.
    h, w = annotated_frame.shape[:2] 
    #görüntünün yeniden boyutlandırıldıktan sonra ulaşabileceği max genişlik ve yükseklik
    scale = min(max_width / w, max_height / h) #görüntünün genişliğinin ne kadar küçültülmesi gerektiği
     #min fonk :iki orandan küçük olanını seçer  > en boy oranını korur                                        #görüntünün yüksekliğinin ne kadar küçültülmesi
    new_w, new_h = int(w * scale), int(h * scale) #int fonk :tam sayı
    resized_frame = cv2.resize(annotated_frame, (new_w, new_h))
    
    # Görüntüyü göster
    cv2.imshow("Instance Segmentation", resized_frame)
    cv2.waitKey(0)

cv2.destroyAllWindows()

# Maskları incele
#results[0] nesnesinin içinde masks adında bir özellik (attribute) var mı ? 
#masks özelliği mevcut olsa bile, içindeki değer boş (None) mu ?
if hasattr(results[0], "masks") and results[0].masks is not None: 
    masks = results[0].masks.data.cpu().numpy()  # [N(sayı), H(yükseklik), W(genişlik)] #maske verilerini işlenebilir bir formata dönüştür
    #veriyi GPU'dan CPU'ya aktarır>verileri NumPy ile kullanabilmek için.
    #numpy fonk: pytorch > numpy 
    print("Mask sayısı:", len(masks)) #tespit edilen ve maskesi oluşturulan nesne sayısı
    if len(masks) > 0:
        print("Bir mask boyutu:", masks[0].shape)
else:
    print("Mask bulunamadı.")





