#$env:KMP_DUPLICATE_LIB_OK="TRUE" :hızlı geçici çözüm >birden fazla OpenMP kopyası yüklense de devam et

import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torchvision import models, transforms
from ultralytics import YOLO
import random


#Cihaz seçimi GPU kontrol

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Kullanılan cihaz: {device}")


# Görsel yükleme

IMG_PATH = "img/image2.jpg"  # Buraya kendi görsel yolunu yaz
original_img = cv2.imread(IMG_PATH)[:, :, ::-1]  # BGR > RGB renk kanallarını tersine çevir 
H, W, _ = original_img.shape

# Semantic Segmentation (DeepLabv3)
#mageNet üzerinde eğitilmiş bu modeli yükle, tahmin moduna (.eval()) al
semantic_model = models.segmentation.deeplabv3_resnet101(pretrained=True).to(device).eval()
#standart ön işleme zinciri
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    #model beklentisi
    transforms.Resize((520, 520)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

input_tensor = preprocess(original_img).unsqueeze(0).to(device)
#modelden tahmini al ve
#çıktının en yüksek olasılıklı sınıf etiketlerini argmax ile seçerek bir sınıf haritası oluştur
with torch.no_grad():
    semantic_output = semantic_model(input_tensor)['out'][0]
semantic_predictions = semantic_output.argmax(0).cpu().numpy()


# Instance Segmentation (YOLOv8)
instance_model = YOLO("yolov8n-seg.pt")
instance_model.to(device)  # GPU kullanımı
#güvenilirlik puanlarını ve segmentasyon maskelerini içeren bir sonuç nesnesi (instance_results) döndür
instance_results = instance_model(IMG_PATH)

# Görselleştirilmiş instance image
#instance_results[0]  analiz edilen ilk görüntünün sonuçları
#.plot(), YOLOv8'e özel bir fonksiyondur.
#tespit edilen tüm nesnelerin üzerine sınırlayıcı kutuları ve renkli segmentasyon maskelerini otomatik olarak çizer
#ve bu görselleştirilmiş görüntüyü döndürür.
instance_img = instance_results[0].plot()[:, :, ::-1]  # BGR > RGB kanalları tersine çevir 

# Maskları numpy array olarak al
instance_masks = [] # maske verilerini saklamak için



#hasattr(obj, "attr"), bir nesnenin (instance_results[0]) belirli bir özelliğe ("masks") sahip olup olmadığını kontrol.
if hasattr(instance_results[0], "masks") and instance_results[0].masks is not None:
    #maske verilerini al ve üzerinde işlem yapmaya uygun hale getir
    # modelin ürettiği ham maske verileri
    #.cpu >numpy için 
    #.numpy:PyTorch> NumPy = her nesne için bir maske [Nesne_Sayısı, Yükseklik, Genişlik] şeklinde 
    instance_masks = instance_results[0].masks.data.cpu().numpy()

print(f"[INFO] Instance Segmentation - Tespit edilen nesne sayısı: {len(instance_masks)}")
if len(instance_masks) > 0:
    print("İlk mask boyutu:", instance_masks[0].shape)

# Mask Overlay (saydam renk)
#YOLOv8'den gelen maskeleri orijinal görüntü üzerine saydam 
overlay_img = original_img.copy()

for mask in instance_masks:
    #520x520 boyutundan cv2.resize() ile orj boyutlarına geri getir
    mask_resized = cv2.resize(mask.astype(np.float32), (W, H))
    #Her bir maskeye random bir renk ata ve 
    #pikselleri orj görüntünün pikselleriyle birleştir
    #saydam bir katman (overlay) oluştur.
    color = [random.randint(0, 255) for _ in range(3)] #örn:[128, 56, 210]
    #olasılığı 0.5'in üzerinde olan pikselleri seç , bu bölgeler overlay_img üzerinde maskenin yerleştirileceği bölgeler
    overlay_img[mask_resized > 0.5] = (
       #görüntü karıştırma (image blending) 
        0.5 * overlay_img[mask_resized > 0.5] + # maskelenen bölgedeki orj piksel değerlerinin yarısını al>bu, saydamlık etkisi için
        0.5 * np.array(color) #oluşturulan rastgele rengin yarısını al
    )

overlay_img = overlay_img.astype(np.uint8)

# Görselleştirme (4 panel)

plt.figure(figsize=(20, 6))

plt.subplot(1, 4, 1)
plt.imshow(original_img)
plt.title("Orijinal Görsel")
plt.axis("off")

plt.subplot(1, 4, 2)
plt.imshow(semantic_predictions, cmap="tab20")
plt.title("Semantic Segmentation (Sınıf Haritası)")
plt.axis("off")

plt.subplot(1, 4, 3)
plt.imshow(instance_img)
plt.title("Instance Segmentation (Mask + BBox)")
plt.axis("off")

plt.subplot(1, 4, 4)
plt.imshow(overlay_img)
plt.title("Mask Overlay (Saydam Renk)")
plt.axis("off")

plt.tight_layout()
plt.show()
