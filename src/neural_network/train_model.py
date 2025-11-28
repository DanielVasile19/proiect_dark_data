import os
import sys
import joblib
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
INPUT_FILE = os.path.join(BASE_DIR, 'data', 'raw', 'rapoarte_mentenanta_v2.csv')
OUTPUT_DIR = os.path.join(BASE_DIR, 'data', 'processed')


def incarca_date():
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"Nu s-a gasit fisierul: {INPUT_FILE}")
    return pd.read_csv(INPUT_FILE)


def antreneaza_model():
    df = incarca_date()

    X_text = df['text_raport'].astype(str).tolist()

    encoder_problema = LabelEncoder()
    y_problema = encoder_problema.fit_transform(df['eticheta_problema'])

    encoder_departament = LabelEncoder()
    y_departament = encoder_departament.fit_transform(df['eticheta_departament'])

    encoder_urgenta = LabelEncoder()
    y_urgenta = encoder_urgenta.fit_transform(df['eticheta_urgenta'])

    num_clase_problema = len(encoder_problema.classes_)
    num_clase_departament = len(encoder_departament.classes_)
    num_clase_urgenta = len(encoder_urgenta.classes_)

    vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), max_features=5000)
    X_vec = vectorizer.fit_transform(X_text)
    X_gata = X_vec.toarray()

    X_train, X_test, y_train_prob, y_test_prob, y_train_dep, y_test_dep, y_train_urg, y_test_urg = train_test_split(
        X_gata, y_problema, y_departament, y_urgenta, test_size=0.2, random_state=42
    )

    inputs = tf.keras.layers.Input(shape=(X_gata.shape[1],))
    x = tf.keras.layers.Dense(128, activation='relu')(inputs)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    body_output = tf.keras.layers.Dropout(0.3)(x)

    output_problema = tf.keras.layers.Dense(num_clase_problema, activation='softmax', name='out_problema')(body_output)
    output_departament = tf.keras.layers.Dense(num_clase_departament, activation='softmax', name='out_departament')(
        body_output)
    output_urgenta = tf.keras.layers.Dense(num_clase_urgenta, activation='softmax', name='out_urgenta')(body_output)

    model = tf.keras.Model(inputs=inputs, outputs=[output_problema, output_departament, output_urgenta])

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

    model.fit(
        X_train,
        {'out_problema': y_train_prob, 'out_departament': y_train_dep, 'out_urgenta': y_train_urg},
        epochs=15,
        batch_size=32,
        validation_data=(X_test,
                         {'out_problema': y_test_prob, 'out_departament': y_test_dep, 'out_urgenta': y_test_urg}),
        verbose=1
    )

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    model.save(os.path.join(OUTPUT_DIR, 'model_dispecer_v2.keras'))
    joblib.dump(vectorizer, os.path.join(OUTPUT_DIR, 'vectorizer_v2.joblib'))
    joblib.dump(encoder_problema, os.path.join(OUTPUT_DIR, 'encoder_problema_v2.joblib'))
    joblib.dump(encoder_departament, os.path.join(OUTPUT_DIR, 'encoder_departament_v2.joblib'))
    joblib.dump(encoder_urgenta, os.path.join(OUTPUT_DIR, 'encoder_urgenta_v2.joblib'))

    print(f"Modele salvate in: {OUTPUT_DIR}")


if __name__ == "__main__":
    antreneaza_model()