# Real-Time Age and Gender Detection

Bu proje, webcam görüntüsü üzerinden **gerçek zamanlı yüz algılama**, **yaş tahmini** ve **cinsiyet sınıflandırması** gerçekleştiren bir bilgisayarlı görü uygulamasıdır.

Proje, önceden eğitilmiş derin öğrenme modellerinin gerçek zamanlı bir sisteme entegre edilmesini (deployment) hedeflemektedir.

---

## 🚀 Özellikler
- Gerçek zamanlı kamera görüntüsü alma
- MediaPipe ile yüz algılama
- CNN tabanlı model ile cinsiyet tahmini (Male / Female)
- Vision Transformer (ViT) tabanlı model ile yaş aralığı tahmini
- Adaptif kullanıcı arayüzü (yüz algılanınca bilgi gösterimi)

---

## 🧠 Kullanılan Teknolojiler
- Python 3.10
- OpenCV
- MediaPipe
- Hugging Face Transformers
- CNN (Caffe tabanlı)
- Vision Transformer (ViT)
- NumPy, PIL

---

## 📊 Kullanılan Modeller
- **Yüz Algılama:** MediaPipe Face Detection  
- **Cinsiyet Tahmini:** Caffe tabanlı CNN modeli (Levi & Hassner)
- **Yaş Tahmini:** Vision Transformer (ViT) tabanlı model  
  (`nateraw/vit-age-classifier`, Adience veri seti ile eğitilmiş)
  👤 Cinsiyet: nateraw/vit-gender-classifier
  🎂 Yaş: nateraw/vit-age-classifier

> Not: Bu projede modeller sıfırdan eğitilmemiş, önceden eğitilmiş (pretrained) modeller kullanılmıştır.

## ⬇️ Download (İndirmek için linklere tıklayın)
pip install transformers torch pillow
-Terminalden kütüphaneyi yükleyin.

https://github.com/spmallick/learnopencv/tree/master/AgeGender
https://raw.githubusercontent.com/spmallick/learnopencv/master/AgeGender/gender_deploy.prototxt

---

## 📂 Proje Yapısı
<img width="373" height="268" alt="Image" src="https://github.com/user-attachments/assets/a66f76b3-4f9a-4d21-b599-696a476a12c4" />

## Sonuçlar 
<img width="590" height="645" alt="Image" src="https://github.com/user-attachments/assets/522bd820-94dd-4029-a3bb-6d91a6f4bfa0" />
<img width="642" height="637" alt="Image" src="https://github.com/user-attachments/assets/3600d9cc-f090-4b22-8e61-a2d06109c0a5" />
<img width="603" height="652" alt="Image" src="https://github.com/user-attachments/assets/3e4b547a-c687-4b57-be7e-bd0dfeaf740f" />
