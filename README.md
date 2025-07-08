# AudioEmotionDetection

Ses Tabanlı Duygu Analizi Projesi

CREMA-D ve EMO-DB veri seti kullanılarak ses tabanlı duygu sınıflandırması yapılmıştır.

## Dosyalar:
- train.ipynb: Veri ön işleme, öznitelik çıkarımı, SMOTE, LightGBM öznitelik seçimi ve model eğitimi. Modeller (LightGBM, SVM, MLP) kaydedilir.
- test.ipynb: Eğitilen modelleri yükler ve test setinde değerlendirir.
- datasets/CREMA-D: Her sınıf için 50 ses dosyasından oluşan veri seti alt kümesi.
  datasets/EMO-DB: Her sınıf için 50 ses dosyasından oluşan veri seti alt kümesi.
- lgbm_model.pkl, svm_model.pkl, mlp_model.pkl: Eğitilen modeller.
- scaler.pkl, pca.pkl, label_encoder.pkl: Ön işleme nesneleri.

## Kullanım:
1. CREMA-D alt kümesini datasets/CREMA-D dizinine yerleştirin.
2. train.ipynb dosyasını çalıştırarak modelleri eğitin ve kaydedin.
3. test.ipynb dosyasını çalıştırarak test sonuçlarını görün.

## Gereksinimler:
- Python 3.8+
- Kütüphaneler: numpy, pandas, librosa, scikit-learn, imbalanced-learn, lightgbm, matplotlib, seaborn
- Kütüphane güncelleme: pip install -U scikit-learn imbalanced-learn
