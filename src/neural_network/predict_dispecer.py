import tensorflow as tf
import joblib
import numpy as np
import sys

print("-------------------------------------------------")
print("🚀 [Pasul 3] Script de Predicție 'Dispecer Inteligent'")
print("-------------------------------------------------")

# --- 1. ÎNCĂRCAREA MODELULUI ȘI A "TRADUCĂTOARELOR" ---
print("Se încarcă modelul Multi-Task și toate componentele V2...")
try:
    model = tf.keras.models.load_model('model_dispecer_v2.keras')
    vectorizer = joblib.load('vectorizer_v2.joblib')
    encoder_problema = joblib.load('encoder_problema_v2.joblib')
    encoder_departament = joblib.load('encoder_departament_v2.joblib')
    encoder_urgenta = joblib.load('encoder_urgenta_v2.joblib')
    print("✅ Toate componentele V2 au fost încărcate.")
except Exception as e:
    print(f"EROARE: Nu am putut încărca fișierele V2.")
    print(f"Asigură-te că ai rulat 'antreneaza_model_MULTITASK.py' și că fișierele există.")
    print(e)
    sys.exit()

# --- 2. TEXTUL NOU (Textul "Murdar" scris de tine) ---
# Joacă-te cu această propoziție!
# Folosește greșeli de scriere și expresii din generatorul V2
text_nou_utilizator = "mtoor ars pe linia 1. Urgent, linia blocata."

print(f"\nSe analizează raportul: '{text_nou_utilizator}'")

# --- 3. PROCESUL DE PREDICȚIE ---

# 1. Transformăm textul nou în vectorul TF-IDF
text_vec = vectorizer.transform([text_nou_utilizator])

# 2. Convertim în formatul "dens" pe care îl vrea Keras
text_gata = text_vec.toarray()

# 3. Facem predicția!
# De data aceasta, 'predictie_bruta' va fi o LISTĂ cu 3 elemente (un array pt fiecare cap)
predictie_bruta = model.predict(text_gata)

# 4. Interpretăm fiecare ieșire (fiecare "cap")
# Ieșirea 0: Problema
prob_problema = predictie_bruta[0][0] # Luăm probabilitățile pentru problemă
idx_problema = np.argmax(prob_problema) # Găsim indexul câștigător
eticheta_problema = encoder_problema.inverse_transform([idx_problema])[0]

# Ieșirea 1: Departament
prob_departament = predictie_bruta[1][0]
idx_departament = np.argmax(prob_departament)
eticheta_departament = encoder_departament.inverse_transform([idx_departament])[0]

# Ieșirea 2: Urgență
prob_urgenta = predictie_bruta[2][0]
idx_urgenta = np.argmax(prob_urgenta)
eticheta_urgenta = encoder_urgenta.inverse_transform([idx_urgenta])[0]

# --- 4. AFIȘAREA REZULTATULUI ---
print("\n--- REZULTATUL ANALIZEI (Dispecer AI) ---")
print(f"Text 'Dark Data':     '{text_nou_utilizator}'")
print("------------------------------------------")
print(f"Problemă Identificată:  {eticheta_problema} (Acuratețe: {prob_problema[idx_problema]*100:.2f}%)")
print(f"Departament Alocat:     {eticheta_departament} (Acuratețe: {prob_departament[idx_departament]*100:.2f}%)")
print(f"Urgență Stabilită:      {eticheta_urgenta} (Acuratețe: {prob_urgenta[idx_urgenta]*100:.2f}%)")