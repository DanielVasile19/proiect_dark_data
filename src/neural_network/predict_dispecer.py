import os
import sys
import joblib
import numpy as np
import tensorflow as tf

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')


def incarca_modele():
    try:
        model = tf.keras.models.load_model(os.path.join(PROCESSED_DIR, 'model_dispecer_v2.keras'))
        vectorizer = joblib.load(os.path.join(PROCESSED_DIR, 'vectorizer_v2.joblib'))
        enc_prob = joblib.load(os.path.join(PROCESSED_DIR, 'encoder_problema_v2.joblib'))
        enc_dep = joblib.load(os.path.join(PROCESSED_DIR, 'encoder_departament_v2.joblib'))
        enc_urg = joblib.load(os.path.join(PROCESSED_DIR, 'encoder_urgenta_v2.joblib'))
        return model, vectorizer, enc_prob, enc_dep, enc_urg
    except Exception as e:
        print(f"Eroare la incarcare modele: {e}")
        sys.exit(1)


def prezice_raport(text):
    model, vectorizer, enc_prob, enc_dep, enc_urg = incarca_modele()

    text_vec = vectorizer.transform([text]).toarray()
    pred = model.predict(text_vec, verbose=0)

    idx_p = np.argmax(pred[0][0])
    idx_d = np.argmax(pred[1][0])
    idx_u = np.argmax(pred[2][0])

    return {
        'problema': enc_prob.inverse_transform([idx_p])[0],
        'conf_p': pred[0][0][idx_p],
        'departament': enc_dep.inverse_transform([idx_d])[0],
        'conf_d': pred[1][0][idx_d],
        'urgenta': enc_urg.inverse_transform([idx_u])[0],
        'conf_u': pred[2][0][idx_u]
    }


if __name__ == "__main__":
    text_test = "mtoor ars pe linia 1. Urgent, linia blocata."
    rezultat = prezice_raport(text_test)

    print("-" * 50)
    print(f"Input: {text_test}")
    print("-" * 50)
    print(f"Problema:    {rezultat['problema']} ({rezultat['conf_p']:.2%})")
    print(f"Departament: {rezultat['departament']} ({rezultat['conf_d']:.2%})")
    print(f"Urgenta:     {rezultat['urgenta']} ({rezultat['conf_u']:.2%})")