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

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
INPUT_FILE = os.path.join(BASE_DIR, 'data', 'raw', 'rapoarte_mentenanta_v2.csv')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
DOCS_DIR = os.path.join(BASE_DIR, 'docs')


def setup_directories():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(DOCS_DIR, exist_ok=True)


def load_data(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    return pd.read_csv(filepath)


def plot_learning_curves(history, save_path):
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training Dynamics')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()


def main():
    setup_directories()

    print(f"Loading data from {INPUT_FILE}...")
    df = load_data(INPUT_FILE)

    X_text = df['text_raport'].astype(str).tolist()

    encoder_prob = LabelEncoder()
    y_prob = encoder_prob.fit_transform(df['eticheta_problema'])

    encoder_dept = LabelEncoder()
    y_dept = encoder_dept.fit_transform(df['eticheta_departament'])

    encoder_urg = LabelEncoder()
    y_urg = encoder_urg.fit_transform(df['eticheta_urgenta'])

    num_classes_prob = len(encoder_prob.classes_)
    num_classes_dept = len(encoder_dept.classes_)
    num_classes_urg = len(encoder_urg.classes_)

    vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), max_features=7000)
    X_vec = vectorizer.fit_transform(X_text)
    X_dense = X_vec.toarray()

    X_train, X_temp, y_p_train, y_p_temp, y_d_train, y_d_temp, y_u_train, y_u_temp = train_test_split(
        X_dense, y_prob, y_dept, y_urg, test_size=0.3, random_state=42, stratify=y_prob
    )

    X_val, X_test, y_p_val, y_p_test, y_d_val, y_d_test, y_u_val, y_u_test = train_test_split(
        X_temp, y_p_temp, y_d_temp, y_u_temp, test_size=0.5, random_state=42, stratify=y_p_temp
    )

    input_layer = Input(shape=(X_dense.shape[1],))
    x = layers.Dense(128, activation='relu')(input_layer)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation='relu')(x)
    shared_layer = layers.Dropout(0.3)(x)

    out_prob = layers.Dense(num_classes_prob, activation='softmax', name='out_problema')(shared_layer)
    out_dept = layers.Dense(num_classes_dept, activation='softmax', name='out_departament')(shared_layer)
    out_urg = layers.Dense(num_classes_urg, activation='softmax', name='out_urgenta')(shared_layer)

    model = Model(inputs=input_layer, outputs=[out_prob, out_dept, out_urg])

    model.compile(
        optimizer='adam',
        loss={
            'out_problema': 'sparse_categorical_crossentropy',
            'out_departament': 'sparse_categorical_crossentropy',
            'out_urgenta': 'sparse_categorical_crossentropy'
        },
        metrics={
            'out_problema': 'accuracy',
            'out_departament': 'accuracy',
            'out_urgenta': 'accuracy'
        }
    )

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
        CSVLogger(os.path.join(RESULTS_DIR, 'training_history.csv'))
    ]

    print("Starting training...")

    history = model.fit(
        X_train,
        {
            'out_problema': y_p_train,
            'out_departament': y_d_train,
            'out_urgenta': y_u_train
        },
        epochs=30,
        batch_size=32,
        validation_data=(
            X_val,
            {
                'out_problema': y_p_val,
                'out_departament': y_d_val,
                'out_urgenta': y_u_val
            }
        ),
        callbacks=callbacks,
        verbose=1
    )

    plot_learning_curves(history, os.path.join(DOCS_DIR, 'loss_curve.png'))

    print("Evaluating on Test Set...")
    preds = model.predict(X_test, verbose=0)

    pred_prob_indices = np.argmax(preds[0], axis=1)

    acc = accuracy_score(y_p_test, pred_prob_indices)
    f1 = f1_score(y_p_test, pred_prob_indices, average='macro')

    metrics = {
        "test_accuracy": float(acc),
        "test_f1_macro": float(f1)
    }

    with open(os.path.join(RESULTS_DIR, 'test_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)

    print(f"Test Accuracy: {acc:.4f}")
    print(f"Test F1 Score: {f1:.4f}")

    print("Saving artifacts...")
    model.save(os.path.join(MODELS_DIR, 'trained_model.h5'))
    joblib.dump(vectorizer, os.path.join(MODELS_DIR, 'vectorizer_v2.joblib'))
    joblib.dump(encoder_prob, os.path.join(MODELS_DIR, 'encoder_problema_v2.joblib'))
    joblib.dump(encoder_dept, os.path.join(MODELS_DIR, 'encoder_departament_v2.joblib'))
    joblib.dump(encoder_urg, os.path.join(MODELS_DIR, 'encoder_urgenta_v2.joblib'))

    print("Training pipeline completed successfully.")


if __name__ == "__main__":
    main()