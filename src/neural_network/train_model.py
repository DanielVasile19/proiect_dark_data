import os
import json
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import layers, Model, Input

# --- CONFIGURARE CAI ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_FILE = os.path.join(BASE_DIR, 'data', 'raw', 'rapoarte_mentenanta_v2.csv')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
DOCS_DIR = os.path.join(BASE_DIR, 'docs')


def setup_mediu():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(DOCS_DIR, exist_ok=True)


def main():
    setup_mediu()
    print("🚀 START ANTRENARE SISTEM DISPECERAT")

    # 1. INCARCARE DATE
    if not os.path.exists(DATA_FILE):
        print(f"❌ EROARE: Nu gasesc fisierul {DATA_FILE}. Ruleaza generare_date_v2.py intai!")
        return

    print("📖 Citire date...")
    try:
        df = pd.read_csv(DATA_FILE, sep='|')  # Separator Pipe
    except Exception as e:
        print(f"Eroare la citire CSV: {e}")
        return

    # 2. ENCODING ETICHETE
    enc_prob = LabelEncoder()
    y_p = enc_prob.fit_transform(df['eticheta_problema'])

    enc_dep = LabelEncoder()
    y_d = enc_dep.fit_transform(df['eticheta_departament'])

    enc_urg = LabelEncoder()
    y_u = enc_urg.fit_transform(df['eticheta_urgenta'])

    # 3. VECTORIZARE TEXT
    print("🔠 Vectorizare text (TF-IDF)...")
    vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4), max_features=3000)
    X = vectorizer.fit_transform(df['text_raport'].astype(str)).toarray()

    # Split Train/Test
    X_train, X_test, yp_train, yp_test, yd_train, yd_test, yu_train, yu_test = train_test_split(
        X, y_p, y_d, y_u, test_size=0.2, random_state=42
    )

    # 4. CONSTRUCTIE MODEL NEURONAL
    print("🧠 Construire Retea Neuronala...")
    input_layer = Input(shape=(X.shape[1],))

    x = layers.Dense(128, activation='relu')(input_layer)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation='relu')(x)

    # 3 Iesiri separate
    out_p = layers.Dense(len(enc_prob.classes_), activation='softmax', name='out_problema')(x)
    out_d = layers.Dense(len(enc_dep.classes_), activation='softmax', name='out_departament')(x)
    out_u = layers.Dense(len(enc_urg.classes_), activation='softmax', name='out_urgenta')(x)

    model = Model(inputs=input_layer, outputs=[out_p, out_d, out_u])

    # --- FIX AICI ---
    # Folosim dictionar pentru metrics ca sa fim expliciti pentru fiecare iesire
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics={
            'out_problema': 'accuracy',
            'out_departament': 'accuracy',
            'out_urgenta': 'accuracy'
        }
    )

    # 5. ANTRENARE
    print("🔥 Se antreneaza modelul...")
    early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    history = model.fit(
        X_train,
        {'out_problema': yp_train, 'out_departament': yd_train, 'out_urgenta': yu_train},
        epochs=15,
        batch_size=32,
        validation_data=(X_test, {'out_problema': yp_test, 'out_departament': yd_test, 'out_urgenta': yu_test}),
        callbacks=[early_stop],
        verbose=1
    )

    # 6. EVALUARE RAPIDA
    print("\n📊 REZULTATE TESTE:")
    preds = model.predict(X_test, verbose=0)

    # Calcul manual acuratete departament
    pred_dep = np.argmax(preds[1], axis=1)
    acc_dep = np.mean(pred_dep == yd_test)
    print(f"✅ Acuratete Departament: {acc_dep:.2%}")

    # Salvare Grafic Acuratete
    plt.figure(figsize=(10, 6))
    # Cheile din history se schimba usor cand folosim dictionar la compile, dar Keras le standardalizeaza de obicei
    # Verificam cheile disponibile
    keys = history.history.keys()
    if 'out_departament_accuracy' in keys:
        plt.plot(history.history['out_departament_accuracy'], label='Train Dep')
        plt.plot(history.history['val_out_departament_accuracy'], label='Test Dep')
    else:
        # Fallback in caz ca TF schimba numele (uneori pune out_departament_acc)
        # Dar cu setarea curenta ar trebui sa fie ok.
        plt.plot(history.history.get('out_departament_accuracy', []), label='Train')

    plt.title('Performanta Model - Departament')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(DOCS_DIR, 'grafic_performanta.png'))
    plt.close()

    # 7. SALVARE FINALA
    print("💾 Salvare artefacte...")
    model.save(os.path.join(MODELS_DIR, 'trained_model.h5'))
    joblib.dump(vectorizer, os.path.join(MODELS_DIR, 'vectorizer_v2.joblib'))
    joblib.dump(enc_prob, os.path.join(MODELS_DIR, 'encoder_problema_v2.joblib'))
    joblib.dump(enc_dep, os.path.join(MODELS_DIR, 'encoder_departament_v2.joblib'))
    joblib.dump(enc_urg, os.path.join(MODELS_DIR, 'encoder_urgenta_v2.joblib'))

    print("\n✅ PREGATIT PENTRU PREZENTARE! Poti rula dashboard-ul.")


if __name__ == "__main__":
    main()