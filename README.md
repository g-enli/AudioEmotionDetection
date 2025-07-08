## Ses Tabanlı Duygu Analizi Projesi (CREMA-D ve EMO-DB)

CREMA-D ve EMO-DB veri setleriyle ses tabanlı duygu sınıflandırması yapılmıştır.

# Dosya Açıklamaları:
- train.ipynb: Veri ön işleme, öznitelik çıkarımı, SMOTE, LightGBM öznitelik seçimi ve model eğitimi. Modeller her veri seti için ayrı kaydedilir.
- test.ipynb: Eğitilen modelleri yükler ve test setinde değerlendirir.
- datasets/CREMA-D: CREMA-D için her sınıf başına 50 ses dosyası.
- datasets/EMO-DB: EMO-DB için her sınıf başına 50 ses dosyası.
- lgbm_model_CREMA-D.pkl, svm_model_CREMA-D.pkl, mlp_model_CREMA-D.pkl: CREMA-D modelleri.
- lgbm_model_EMO-DB.pkl, svm_model_EMO-DB.pkl, mlp_model_EMO-DB.pkl: EMO-DB modelleri.
- scaler_CREMA-D.pkl, pca_CREMA-D.pkl, label_encoder_CREMA-D.pkl: CREMA-D ön işleme nesneleri.
- scaler_EMO-DB.pkl, pca_EMO-DB.pkl, label_encoder_EMO-DB.pkl: EMO-DB ön işleme nesneleri.
- X_test_selected_CREMA-D.npy, y_test_CREMA-D.npy: CREMA-D test verisi.
- X_test_selected_EMO-DB.npy, y_test_EMO-DB.npy: EMO-DB test verisi.

# Kullanım:
1. CREMA-D ve EMO-DB alt kümelerini datasets/ dizinine yerleştirin.
2. train.ipynb dosyasını çalıştırarak modelleri eğitin ve kaydedin.
3. test.ipynb dosyasını çalıştırarak test sonuçlarını görün.

# Gereksinimler:
- Python 3.8+
- Kütüphaneler: numpy, pandas, librosa, scikit-learn, imbalanced-learn, lightgbm, matplotlib, seaborn
- Kütüphane güncelleme: pip install -U scikit-learn imbalanced-learn
