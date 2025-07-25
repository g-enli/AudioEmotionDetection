# Ses Tabanlı Duygu Analizi Projesi (CREMA-D ve EMO-DB)

Bu proje, **CREMA-D** ve **EMO-DB** veri setlerini kullanarak ses tabanlı duygu sınıflandırması yapar. İki farklı yöntem uygulanmıştır:

Datasetlerin orijinal hallerinde bulunduğu [drive linki](https://drive.google.com/drive/folders/1hWX3PMePLLoGJEq0iMiUh6-3zWsqQ0sJ)

1.  **Yöntem 1:** LightGBM, SVM ve MLP tabanlı makine öğrenimi modelleriyle duygu sınıflandırması (`train_ml.ipynb`, `test_ml.ipynb`).
2.  **Yöntem 2:** LSTM tabanlı tekrarlayan sinir ağı (RNN) modeli ile duygu sınıflandırması (`train_rnn.ipynb`, `test_rnn.ipynb`).



## Veri Setleri

-   **CREMA-D:** ~7,500 ses dosyası, 6 duygu sınıfı (Happy, Sad, Angry, Fear, Disgust, Neutral). Her sınıf için ~1,250 örnek.
-   **CREMA-D_AUG:** CREMA-D veri setine veri artırma (gürültü ekleme, hız değiştirme, perde kaydırma) uygulanarak ~30,000 örneğe genişletildi (4 kat artış).
-   **EMO-DB:** ~25,000 ses dosyası, 7 duygu sınıfı (Angry, Happy, Disgust, Fear, Surprised, Sad, Neutral). Her sınıf için ~3,500 örnek.



## Yöntemler

### Yöntem 1: Makine Öğrenimi Modelleri

-   **Öznitelik Çıkarımı:** `librosa` ile MFCC, kroma, mel spektrogramı gibi ses öznitelikleri çıkarılır.
-   **Ön İşleme:** `StandardScaler` ile ölçeklendirme, `SMOTE` ile sınıf dengesizliği çözümü.
-   **Modeller:** LightGBM, SVM ve MLP modelleri eğitilir.
-   **Değerlendirme:** Test setinde doğruluk, F1-skoru, hassasiyet ve duyarlılık metrikleri hesaplanır.

### Yöntem 2: LSTM Tabanlı RNN

-   **Veri Hazırlama:** Ses dosyalarından 13 MFCC katsayısı librosa ile zaman serisi formatında ([örnek_sayısı, zaman_adımları, 13]) hazırlanır. Ses dosyaları sabit uzunluğa (32000 örnek) getirilir ve CREMA-D_AUG için veri artırma (gürültü ekleme, hız değiştirme, perde kaydırma) uygulanır.
-   **Ön İşleme:** `StandardScaler` ile ölçeklendirme, `SMOTE` ile sınıf dengesizliği çözümü (opsiyonel).
-   **Model:** PyTorch ile 2-3 katmanlı LSTM modeli, dropout (0.4-0.5) ve early stopping (patience=5-7) uygulanır.
-   **Değerlendirme:** Eğitim/doğrulama kayıp eğrileri, karışıklık matrisi, doğruluk, F1-skoru, hassasiyet ve duyarlılık grafikleri.



## Dosya Açıklamaları

### Yöntem 1 (Makine Öğrenimi)

-   `train_ml.ipynb`: Veri ön işleme, öznitelik çıkarımı, SMOTE, model eğitimi (LightGBM, SVM, MLP) ve kaydedilmesi.
-   `test_ml.ipynb`: Eğitilmiş modelleri yükler ve test setinde değerlendirir. Sonuçlar metriklerle raporlanır.
-   `lgbm_model_*.pkl`, `svm_model_*.pkl`, `mlp_model_*.pkl`: Her veri seti için eğitilmiş modeller.
-   `label_encoder_*.pkl`: Veri setlerine göre etiket kodlayıcılar.
-   `X_test_selected_*.npy`, `y_test_*.npy`: Her veri seti için test verileri.

### Yöntem 2 (LSTM Tabanlı RNN)

-   `train_rnn.ipynb`: Veri ön işleme, MFCC çıkarımı, SMOTE (opsiyonel), LSTM modeli eğitimi ve değerlendirmesi.
-   `test_rnn.ipynb`: Eğitilmiş LSTM modellerini yükler ve test setinde değerlendirir.
-   `rnn_model_*.pth`, `rnn_model_*.pkl`: Her veri seti için eğitilmiş LSTM modeli (PyTorch state\_dict ve tam model).
-   `scaler_rnn_*.pkl`, `label_encoder_rnn_*.pkl`: Her veri seti için ölçekleyici ve etiket kodlayıcı.
-   `X_test_rnn_*.npy`, `y_test_rnn_*.npy`: Her veri seti için test verileri.



## Kullanım Talimatları

### Ortak Adımlar

1.  **Veri Setlerini Hazırlayın:**
    -   CREMA-D ve EMO-DB veri setlerini `datasets/CREMA-D` ve `datasets/EMO-DB` dizinlerine yerleştirin.
    -   CREMA-D dosyalarının dosya adlarında duygu kodları olduğundan emin olun (örn: `1001_DFA_HAP_XX.wav`).
    -   EMO-DB dosyalarının duygu sınıflarına göre alt klasörlerde olduğundan emin olun (örn: `datasets/EMO-DB/angry/`).

2.  **Gereksinimleri Yükleyin:**
    ```bash
    pip install -U numpy pandas librosa scikit-learn imbalanced-learn lightgbm matplotlib seaborn torch joblib
    ```

3.  **Google Drive Bağlantısı (Colab için):**
    -   Notebook dosyalarını (`train_ml.ipynb`, `test_ml.ipynb`, `train_rnn.ipynb`, `rnn_train.ipynb`) Google Drive’a yükleyin.
    -   Veri setlerini `/content/drive/MyDrive/AudioEmotionDetection/datasets/` altına yerleştirin.
    -   Notebook’larda Google Drive’ı bağlayın:
        ```python
        from google.colab import drive
        drive.mount('/content/drive')
        ```

### Yöntem 1: Makine Öğrenimi

1.  **`train_ml.ipynb` dosyasını çalıştırın:** Veri ön işleme, öznitelik çıkarımı, SMOTE, model eğitimi ve kaydetme işlemleri yapılır.
2.  **`test_ml.ipynb` dosyasını çalıştırın:** Kaydedilen modeller yüklenir ve test setinde değerlendirilerek sonuçlar raporlanır.

### Yöntem 2: LSTM Tabanlı RNN

1.  **`train_rnn.ipynb` dosyasını çalıştırın:** Veri ön işleme, MFCC çıkarımı, model eğitimi ve değerlendirmesi yapılır. Her veri seti için ayrı modeller eğitilir ve kaydedilir.
2.  **`test_rnn.ipynb` dosyasını çalıştırın:** Kaydedilen LSTM modelleri yüklenir ve test setinde değerlendirilir.

**Çıktılar:** Modeller `.pth` ve `.pkl` formatında, test verileri, scaler'lar ve etiket kodlayıcılar `.npy` ve `.pkl` formatında kaydedilir.



## Gereksinimler

-   **Python:** 3.8 veya üstü
-   **Kütüphaneler:**
    ```bash
    pip install -U numpy pandas librosa scikit-learn imbalanced-learn lightgbm matplotlib seaborn torch joblib
    ```
-   **Donanım (Yöntem 2 için önerilen):**
    -   GPU (CUDA destekli, PyTorch için).
    -   Minimum 15 GB RAM (özellikle CREMA-D_AUG ve EMO-DB için), Colab Pro öneririm.

---

## Notlar

-   **Veri Seti Boyutları:** `CREMA-D_AUG` veri seti, artırma ile ~30,000 örneğe ulaştığı için eğitim süresini artırabilir.
-   **Hata Ayıklama:** Dosya yollarının (`datasets/CREMA-D`, `datasets/EMO-DB`) doğru olduğundan emin olun.
-   **Performans İyileştirme:**
    -   **Yöntem 2:** LSTM modelinde `hidden_size=128`, `num_layers=3` veya `bidirectional LSTM` denenebilir.
    -   **Yöntem 1:** LightGBM için hiperparametre optimizasyonu (`GridSearchCV`) yapılabilir.
