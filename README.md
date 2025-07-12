# Ses Tabanlı Duygu Analizi Projesi (CREMA-D ve EMO-DB)

Bu proje, **CREMA-D** ve **EMO-DB** veri setleri kullanılarak gerçekleştirilmiş bir ses tabanlı duygu tanıma (SER - Speech Emotion Recognition) çalışmasıdır. Projede, model performansını artırmak amacıyla CREMA-D veri seti için **veri artırma (data augmentation)** teknikleri de uygulanmıştır.

(veri setlerinin orijinal hallerinin de bulunduğu)[drive linki](https://drive.google.com/drive/folders/1hWX3PMePLLoGJEq0iMiUh6-3zWsqQ0sJ?usp=drive_link)

LightGBM, SVM ve MLP (Multi-Layer Perceptron) olmak üzere üç farklı makine öğrenmesi modeli her bir veri seti (CREMA-D, CREMA-D Artırılmış, EMO-DB) için ayrı ayrı eğitilmiş ve değerlendirilmiştir.

## Dosya Açıklamaları

-   `train.ipynb`: Veri ön işleme, öznitelik çıkarımı, veri artırma (augmentation), model eğitimi, değerlendirme ve sonuçların kaydedilmesi gibi tüm adımları içeren ana Jupyter Notebook dosyasıdır.
-   `test.ipynb`: Daha önceden eğitilip kaydedilmiş olan modelleri (`.pkl` dosyaları) yükleyerek test verisi üzerinde hızlı tahminler ve değerlendirmeler yapmak için kullanılır.

### Modeller (`.pkl`)

-   **CREMA-D Modelleri (Orijinal Veri Seti):**
    -   `lgbm_model_CREMA-D.pkl`: LightGBM modeli
    -   `svm_model_CREMA-D.pkl`: Destek Vektör Makineleri (SVM) modeli
    -   `mlp_model_CREMA-D.pkl`: Çok Katmanlı Algılayıcı (MLP) modeli

-   **CREMA-D Modelleri (Artırılmış Veri Seti - AUG):**
    -   `lgbm_model_CREMA-D_AUG.pkl`: Veri artırma uygulanmış veri seti ile eğitilmiş LightGBM modeli
    -   `svm_model_CREMA-D_AUG.pkl`: Veri artırma uygulanmış veri seti ile eğitilmiş SVM modeli
    -   `mlp_model_CREMA-D_AUG.pkl`: Veri artırma uygulanmış veri seti ile eğitilmiş MLP modeli

-   **EMO-DB Modelleri:**
    -   `lgbm_model_EMO-DB.pkl`: LightGBM modeli
    -   `svm_model_EMO-DB.pkl`: SVM modeli
    -   `mlp_model_EMO-DB.pkl`: MLP modeli

### Test Verileri ve Etiket Kodlayıcılar

-   `X_test_selected_*.npy`: Modellerin değerlendirilmesi için kullanılan, öznitelik seçimi yapılmış test veri setleri.
-   `y_test_*.npy`: Test veri setlerine karşılık gelen gerçek duygu etiketleri.
-   `label_encoder_*.pkl`: Duygu etiketlerini sayısal formata dönüştürmek için kullanılan `LabelEncoder` nesneleri.

## Kullanım

1.  Proje için kullanılacak olan CREMA-D ve EMO-DB ses dosyalarını içeren veri setlerini hazırlayın ve `train.ipynb` içerisinde belirtilen yollara yerleştirin.
2.  `train.ipynb` not defterini çalıştırarak tüm süreci (veri işleme, eğitim, modellerin ve test verilerinin kaydedilmesi) baştan sona yürütün.
3.  Alternatif olarak, yalnızca kayıtlı modelleri test etmek isterseniz `test.ipynb` dosyasını çalıştırarak sonuçları görüntüleyebilirsiniz.

## Gereksinimler

-   Python 3.8+
-   Gerekli Kütüphaneler: `numpy`, `pandas`, `librosa`, `scikit-learn`, `imbalanced-learn`, `lightgbm`, `matplotlib`, `seaborn`

Kütüphaneleri aşağıdaki komut ile kurabilir veya güncelleyebilirsiniz:
```bash
pip install -U numpy pandas librosa scikit-learn imbalanced-learn lightgbm matplotlib seaborn
