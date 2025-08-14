 # $env:KMP_DUPLICATE_LIB_OK="TRUE"


import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torchvision import models, transforms

# Cihaz seçimi
#gpu kontrol 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Kullanılan cihaz: {device}")

# Modeli yükle ve GPUya taşı
#pretrained=True >  modelin ImageNet veri seti üzerinde önceden eğitilmiş ağırlıklarını kullanmasını sağlıyor.
#Bu sayede, model sıfırdan eğitilmeye gerek kalmadan çok iyi bir performans sergileyebiliyor.
#eval(): Modeli değerlendirme (tahmin) moduna al
#eğitim sırasında kullanılan bazı özel katmanları (dropout vb) devre dışı bırakarak tutarlı sonuçlar elde etmemi sağlar
model = models.segmentation.deeplabv3_resnet101(pretrained=True).to(device).eval()

# Görüntü ön işleme
preprocess = transforms.Compose([
    #opencv > to pyhon ımage library 
    transforms.ToPILImage(),
    #Modelin giriş boyutu 520x520 piksel olduğu için görüntüyü bu boyuta yeniden boyutlandır
    transforms.Resize((520, 520)),
    #örüntü piksellerini 0-255 aralığından 0-1 aralığına normalize ediyor ve PyTorch tensor formatına dönüştür
    transforms.ToTensor(),
    #ransforms.Normalize:modelin eğitiminde kullanılan ortalama ve standart sapma değerlerine göre
    #tensörün piksellerini normalize ediyor
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]) 
                         #modelin eğitiminde kullanılan ImageNet veri setinin istatistiksel özellikleridir.
                         #Bu değerler, modelin tutarlı ve doğru tahminler yapabilmesi için "evrensel dil" görevi görür.
])

# Görsel oku ve hazırla
img = cv2.imread("img/image.jpg")[:, :, ::-1]  #BGR >  RGB kanalları tersine çevir 
input_tensor = preprocess(img).unsqueeze(0).to(device) #tek bir görüntü verdiğimiz için > unsqueeze

# Tahmin
#hem bellek kullanımını azaltır hem de hesaplamaları hızlandırır> with torch.no_grad()
with torch.no_grad():
    #model(input_tensor)['out'][0]: Modelin tahminini al
    #DeepLabV3, çıktı olarak bir sözlük (dict) döndürür ve sonuç 'out' anahtarının içinde 
    output = model(input_tensor)['out'][0]
    #argmax fonksiyonu, her piksel için en yüksek olasılığa sahip olan sınıfın indeksini (etiketini) seçer.
output_predictions = output.argmax(0).cpu().numpy() #eriyi CPU'ya taşıyıp NumPy yap

# Sınıf haritası görselleştir
plt.imshow(output_predictions) #NumPy dizisindeki her bir sınıf etiketini (örneğin, 0, 1, 2, ...) farklı bir renge dönüştür
plt.title("Semantic Segmentation - Sınıf Haritası")
plt.axis("off")
plt.show()
