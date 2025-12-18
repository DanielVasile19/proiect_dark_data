import os
import sys
import joblib
import numpy as np
import tensorflow as tf

# Configurare cai
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR = os.path.join(BASE_DIR, 'models')

def incarca_artefacte():
    try:
        # Incarcare resurse antrenate
        model = tf.keras.models.load_model(os.path.join(MODELS_DIR, 'trained_model.h5'))
        vect = joblib.load(os.path.join(MODELS_DIR, 'vectorizer_v2.joblib'))
        enc_p = joblib.load(os.path.join(MODELS_DIR, 'encoder_problema_v2.joblib'))
        enc_d = joblib.load(os.path.join(MODELS_DIR, 'encoder_departament_v2.joblib'))
        enc_u = joblib.load(os.path.join(MODELS_DIR, 'encoder_urgenta_v2.joblib'))
        return model, vect, enc_p, enc_d, enc_u
    except Exception as e:
        print(f"Eroare incarcare modele: {e}")
        sys.exit(1)

def prezice(text):
    model, vectorizer, enc_p, enc_d, enc_u = incarca_artefacte()
    
    # Inferenta
    vec = vectorizer.transform([text]).toarray()
    pred = model.predict(vec, verbose=0)
    
    # Decodare
    i_p, i_d, i_u = np.argmax(pred[0]), np.argmax(pred[1]), np.argmax(pred[2])
    
    return {
        'prob': enc_p.inverse_transform([i_p])[0],
        'dep': enc_d.inverse_transform([i_d])[0],
        'urg': enc_u.inverse_transform([i_u])[0],
        'conf': np.max(pred[0])
    }

if __name__ == "__main__":
    # Testare
    sample = "nu avem internet la plc"
    res = prezice(sample)
    
    print(f"\nINPUT: {sample}")
    print(f"REZULTAT: {res['prob']} | {res['dep']} | {res['urg']} (Conf: {res['conf']:.2%})")