# 🍅 LeafLens - Tomato Leaf Disease Detection with EfficientNetB3
[EN]
LeafLens, developed as part of a computer engineering graduation project and experimented with during that process, is a deep learning-based system aiming to classify tomato leaf diseases using image data.

## Dataset

This project utilizes the [Tomato Disease Multiple Sources Dataset](https://www.kaggle.com/datasets/cookiefinder/tomato-disease-multiple-sources) from Kaggle. Due to its size and licensing restrictions, the dataset is **not included** in this repository.

Please download the dataset from the official source and follow the instructions in data_preparation.md to prepare the data for training and testing.
## Dataset Arrangement

The dataset used in this project is originally the Tomato Disease Multiple Sources Dataset, from which a balanced subset was created for this project.

The training (train) folder contains 1001 images per disease class.
The testing (test) folder includes 445 images per class.
The validation (val) folder is organized with 501 images per class.
This ensures that each class is equally represented in the training, testing, and validation phases, minimizing class imbalance.

## Project Objective
- Diagnose tomato leaf diseases using CNNs (EfficientNetB3).
- Visualize model attention using Grad-CAM.
- Assist in agricultural decision-making with AI-powered diagnostics.

## Project Structure
| File.                 | Description                                      |
|-----------------------|--------------------------------------------------|
| `train_model.py`      | Trains the model with training/validation data.  |
| `evaluate_model.py`   | Evaluates model and plots results.               |
| `predict_external.py` | Predicts disease from an external leaf image.    |
| `gradcam_utils.py`    | Applies segmentation and Grad-CAM heatmaps.      |

## How to Run

Model File

The model file has been uploaded to the repository using Git LFS.

To clone the repository completely and correctly, run:
git lfs install
git clone https://github.com/bettekren/EfficientNetB3-TomatoDisease.git


Install dependencies:
```bash
pip install -r requirements.txt

| Epoch | Train Accuracy | Val Accuracy | Train Loss | Val Loss |
| ----- | -------------- | ------------ | ---------- | -------- |
| 1     | 0.0938         | 0.1000       | 3.1871     | 2.3915   |
| 2     | 0.3975         | 0.4956       | 1.8266     | 1.5057   |
| 3     | 0.6580         | 0.7004       | 1.0568     | 1.0595   |
| 4     | 0.7624         | 0.8266       | 0.7784     | 0.6582   |
| 5     | 0.8372         | 0.7850       | 0.5845     | 0.8354   |
| 6     | 0.8701         | 0.9130       | 0.4838     | 0.3836   |
| 7     | 0.9000         | 0.9358       | 0.4021     | 0.2803   |
| 8     | 0.9149         | 0.9320       | 0.3490     | 0.2848   |
| 9     | 0.9263         | 0.7672       | 0.3079     | 0.8067   |
| 10    | 0.9369         | 0.9154       | 0.2739     | 0.3480   |

| Metric                | Value     |
| --------------------- | --------- |
| Test Accuracy         | 95.3%     |
| Test Loss             | 0.2393    |
| Training Accuracy     | 93.7%     |
| Validation Accuracy   | 91.5%     |
| Test Samples          | 3211      |
| Batch Size            | 1         |
| Total Batches         | 3211      |

Results
 Training Accuracy: 93.6%

 Validation Accuracy: 91.5%

 Test Accuracy: 95.3%

Betül Ekren
Computer Engineering, ZBEÜ - Class of 2025
Project developed as part of final year thesis.

[TR]
# LeafLens - EfficientNetB3 ile Domates Yaprağı Hastalık Tespiti
LeafLens, bilgisayar mühendisliği mezuniyet projesi kapsamında geliştirilmiş ve proje sürecinde denenmiş,
görüntü verisi kullanarak domates yaprağı hastalıklarını sınıflandırmayı amaçlayan derin öğrenme tabanlı bir sistemdir.

## Projenin Amacı
- CNN’ler (EfficientNetB3) kullanarak domates yaprağı hastalıklarının teşhisi.
- Modelin dikkat ettiği alanların Grad-CAM ile görselleştirilmesi.
- Yapay zeka destekli teşhislerle tarımsal karar verme süreçlerine katkı sağlamak.

## Proje Dosya Yapısı
| Dosya                 | Açıklama                                                    |
|-----------------------|-------------------------------------------------------------|
| `train_model.py`      | Eğitim/doğrulama verisi ile modelin eğitilmesi.             |
| `evaluate_model.py`   | Modelin değerlendirilmesi ve sonuçların grafikle gösterimi. |
| `predict_external.py` | Harici yaprak görüntüsünden hastalık tahmini.               |
| `gradcam_utils.py`    | Segmentasyon ve Grad-CAM ısı haritalarının uygulanması.     |

## Çalıştırma

Gerekli paketleri yükleyin:  
```bash
pip install -r requirements.txt


| Epoch | Eğitim Doğruluğu| Doğrulama Doğruluğu | Eğitim Kaybı | Doğrulama Kaybı |
| ----- | --------------  | ------------------- | ------------ | --------------- |
| 1     | 0.0938          | 0.1000              | 3.1871       | 2.3915          |
| 2     | 0.3975          | 0.4956              | 1.8266       | 1.5057          |
| 3     | 0.6580          | 0.7004              | 1.0568       | 1.0595          |
| 4     | 0.7624          | 0.8266              | 0.7784       | 0.6582          |
| 5     | 0.8372          | 0.7850              | 0.5845       | 0.8354          |
| 6     | 0.8701          | 0.9130              | 0.4838       | 0.3836          |
| 7     | 0.9000          | 0.9358              | 0.4021       | 0.2803          |
| 8     | 0.9149          | 0.9320              | 0.3490       | 0.2848          |
| 9     | 0.9263          | 0.7672              | 0.3079       | 0.8067          |
| 10    | 0.9369          | 0.9154              | 0.2739       | 0.3480          |


| Metrik               | Değer     |
| -------------------  | --------- |
| Test Doğruluğu       | 95.3%     |
| Test Kaybı           | 0.2393    |
| Eğitim Doğruluğu     | 93.7%     |
| Doğrulama Doğruluğu  | 91.5%     |
| Test Örnek Sayısı    | 3211      |
| Batch Boyutu         | 1         |
| Total Batch Sayısı   | 3211      |

Sonuçlar
Eğitim Doğruluğu: %93.6
Doğrulama Doğruluğu: %91.5
Test Doğruluğu: %95.3

## Model Dosyası

Model dosyası `Git LFS` kullanılarak repoya yüklenmiştir.

Repoyu tam ve doğru şekilde klonlamak için:

```bash
git lfs install
git clone https://github.com/bettekren/EfficientNetB3-TomatoDisease.git

Betül Ekren
Bilgisayar Mühendisliği, ZBEÜ - 2025 Mezunu
Proje bitirme tezi kapsamında geliştirilmiştir.
