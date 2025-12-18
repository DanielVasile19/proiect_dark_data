import os
import json
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from tensorflow.keras.callbacks import EarlyStopping, CSVLogger
from tensorflow.keras import layers, Model, Input

# Configurare cai
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
INPUT_FILE = os.path.join(BASE_DIR, 'data', 'raw', 'rapoarte_mentenanta_v2.csv')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
DOCS_DIR = os.path.join(BASE_DIR, 'docs')

def setup_mediu():
    # Creare directoare necesare
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(DOCS_DIR, exist_ok=True)

def plot_grafic_loss(history, save_path):
    # Generare curba invatare
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Dinamica Antrenare')
    plt.xlabel('Epoci')
    plt.ylabel('Eroare (Loss)')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def main():
    setup_mediu()
    
    # Incarcare date
    print(f"Incarcare dataset: {INPUT_FILE}")
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError("Lipsa fisier CSV!")
    df = pd.read_csv(INPUT_FILE)
    
    # Preprocesare etichete
    enc_prob = LabelEncoder()
    y_prob = enc_prob.fit_transform(df['eticheta_problema'])
    
    enc_dept = LabelEncoder()
    y_dept = enc_dept.fit_transform(df['eticheta_departament'])
    
    enc_urg = LabelEncoder()
    y_urg = enc_urg.fit_transform(df['eticheta_urgenta'])
    
    # Vectorizare text
    print("Vectorizare TF-IDF...")
    vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), max_features=7000)
    X_dense = vectorizer.fit_transform(df['text_raport'].astype(str)).toarray()
    
    # Split date (Train / Val / Test)
    X_train, X_temp, y_p_train, y_p_temp, y_d_train, y_d_temp, y_u_train, y_u_temp = train_test_split(
        X_dense, y_prob, y_dept, y_urg, test_size=0.3, random_state=42, stratify=y_prob
    )
    
    X_val, X_test, y_p_val, y_p_test, y_d_val, y_d_test, y_u_val, y_u_test = train_test_split(
        X_temp, y_p_temp, y_d_temp, y_u_temp, test_size=0.5, random_state=42, stratify=y_p_temp
    )
    
    # Arhitectura Model Multi-Task
    input_layer = Input(shape=(X_dense.shape[1],))
    x = layers.Dense(128, activation='relu')(input_layer)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation='relu')(x)
    shared = layers.Dropout(0.3)(x)
    
    out_prob = layers.Dense(len(enc_prob.classes_), activation='softmax', name='out_problema')(shared)
    out_dept = layers.Dense(len(enc_dept.classes_), activation='softmax', name='out_departament')(shared)
    out_urg = layers.Dense(len(enc_urg.classes_), activation='softmax', name='out_urgenta')(shared)
    
    model = Model(inputs=input_layer, outputs=[out_prob, out_dept, out_urg])
    
    # Compilare
    model.compile(
        optimizer='adam',
        loss={'out_problema': 'sparse_categorical_crossentropy', 
              'out_departament': 'sparse_categorical_crossentropy', 
              'out_urgenta': 'sparse_categorical_crossentropy'},
        metrics={'out_problema': 'accuracy', 
                 'out_departament': 'accuracy', 
                 'out_urgenta': 'accuracy'}
    )
    
    # Antrenare cu Early Stopping
    print("Start antrenare...")
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
        CSVLogger(os.path.join(RESULTS_DIR, 'training_history.csv'))
    ]
    
    history = model.fit(
        X_train,
        {'out_problema': y_p_train, 'out_departament': y_d_train, 'out_urgenta': y_u_train},
        epochs=30,
        batch_size=32,
        validation_data=(X_val, {'out_problema': y_p_val, 'out_departament': y_d_val, 'out_urgenta': y_u_val}),
        callbacks=callbacks,
        verbose=1
    )
    
    plot_grafic_loss(history, os.path.join(DOCS_DIR, 'loss_curve.png'))
    
    # Evaluare Test Set
    print("Evaluare performanta...")
    preds = model.predict(X_test, verbose=0)
    pred_indices = np.argmax(preds[0], axis=1) # Evaluam pe 'problema'
    
    metrics = {
        "test_accuracy": float(accuracy_score(y_p_test, pred_indices)),
        "test_f1_macro": float(f1_score(y_p_test, pred_indices, average='macro'))
    }
    
    # Salvare rezultate
    with open(os.path.join(RESULTS_DIR, 'test_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
        
    print(f"Accuracy: {metrics['test_accuracy']:.4f} | F1: {metrics['test_f1_macro']:.4f}")
    
    # Salvare artefacte
    print("Salvare modele...")
    model.save(os.path.join(MODELS_DIR, 'trained_model.h5'))
    joblib.dump(vectorizer, os.path.join(MODELS_DIR, 'vectorizer_v2.joblib'))
    joblib.dump(enc_prob, os.path.join(MODELS_DIR, 'encoder_problema_v2.joblib'))
    joblib.dump(enc_dept, os.path.join(MODELS_DIR, 'encoder_departament_v2.joblib'))
    joblib.dump(enc_urg, os.path.join(MODELS_DIR, 'encoder_urgenta_v2.joblib'))

if __name__ == "__main__":
    main()