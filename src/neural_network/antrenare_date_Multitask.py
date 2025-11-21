import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import numpy as np
import sys

print("-------------------------------------------------")
print("🚀 [Pasul 2.v2] Antrenare Model Multi-Task (Dispecer)")
print("-------------------------------------------------")

# --- 1. ÎNCĂRCAREA DATELOR v2 ---
NUME_FISIER_CSV = "rapoarte_mentenanta_v2.csv" # Folosim noul fișier!
try:
    df = pd.read_csv(NUME_FISIER_CSV)
    print(f"✅ Datele V2 '{NUME_FISIER_CSV}' au fost încărcate. ({len(df)} rânduri)")
except FileNotFoundError:
    print(f"EROARE: Fișierul '{NUME_FISIER_CSV}' nu a fost găsit! Rulează mai întâi 'genereaza_date_v2.py'.")
    sys.exit()

# --- 2. PREGĂTIREA X și y (Multiplu) ---
X_text = df['text_raport'].tolist()

# Acum avem 3 ținte (y) separate
encoder_problema = LabelEncoder()
y_problema = encoder_problema.fit_transform(df['eticheta_problema'])
num_clase_problema = len(encoder_problema.classes_)

encoder_departament = LabelEncoder()
y_departament = encoder_departament.fit_transform(df['eticheta_departament'])
num_clase_departament = len(encoder_departament.classes_)

encoder_urgenta = LabelEncoder()
y_urgenta = encoder_urgenta.fit_transform(df['eticheta_urgenta'])
num_clase_urgenta = len(encoder_urgenta.classes_)

print(f"✅ Ținte definite: {num_clase_problema} probleme, {num_clase_departament} departamente, {num_clase_urgenta} urgențe.")

# --- 3. VECTORIZAREA (TF-IDF) ---
print("Se creează 'vocabularul' TF-IDF...")
vectorizer = TfidfVectorizer(max_features=2000) # Am mărit numărul de cuvinte
X_vec = vectorizer.fit_transform(X_text)
X_gata = X_vec.toarray()
print(f"✅ Datele text au fost vectorizate. (Vectori de {X_gata.shape[1]} features)")

# --- 4. ÎMPĂRȚIREA DATELOR ---
# Trebuie să împărțim X și toate cele 3 ținte (y)
X_train, X_test, y_train_prob, y_test_prob, y_train_dep, y_test_dep, y_train_urg, y_test_urg = train_test_split(
    X_gata, y_problema, y_departament, y_urgenta, test_size=0.2, random_state=42)
print(f"✅ Datele au fost împărțite: {len(X_train)} antrenare, {len(X_test)} testare.")


# --- 5. CONSTRUIREA MODELULUI MULTI-TASK ---
print("Se construiește rețeaua neuronală Multi-Task...")
dimensiune_input = X_gata.shape[1]

# 1. Intrarea (Input)
inputs = tf.keras.layers.Input(shape=(dimensiune_input,))

# 2. "Corpul" (Body) - comun pentru toate sarcinile
# Rețeaua învață să înțeleagă textul o singură dată
x = tf.keras.layers.Dense(128, activation='relu')(inputs)
x = tf.keras.layers.Dropout(0.3)(x)
x = tf.keras.layers.Dense(64, activation='relu')(x)
body_output = tf.keras.layers.Dropout(0.3)(x)

# 3. "Capetele" (Heads) - câte un cap de ieșire pentru fiecare sarcină
# Capul 1: Prezice Problema
output_problema = tf.keras.layers.Dense(num_clase_problema, activation='softmax', name='out_problema')(body_output)

# Capul 2: Prezice Departament
output_departament = tf.keras.layers.Dense(num_clase_departament, activation='softmax', name='out_departament')(body_output)

# Capul 3: Prezice Urgența
output_urgenta = tf.keras.layers.Dense(num_clase_urgenta, activation='softmax', name='out_urgenta')(body_output)

# 4. Asamblăm Modelul
# Îi spunem lui Keras: O intrare, trei ieșiri
model = tf.keras.Model(
    inputs=inputs,
    outputs=[output_problema, output_departament, output_urgenta]
)

# 5. Compilarea Modelului
# Trebuie să specificăm o funcție de eroare (loss) pentru FIECARE cap
# NOU (Corect)
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

model.summary()

# --- 6. ANTRENAREA MODELULUI ---
print("\n--- Începe Antrenarea Multi-Task ---")
# La 'y' (țintă), îi dăm un dicționar care potrivește ieșirile
history = model.fit(
    X_train,
    {'out_problema': y_train_prob, 'out_departament': y_train_dep, 'out_urgenta': y_train_urg},
    epochs=15, # Antrenăm puțin mai mult
    batch_size=32,
    validation_data=(X_test, {'out_problema': y_test_prob, 'out_departament': y_test_dep, 'out_urgenta': y_test_urg}),
    verbose=1
)

print("✅ Antrenarea este finalizată!")

# --- 7. EVALUAREA FINALĂ ---
print("\n--- Evaluarea Performanței Finale pe Setul de Test ---")
scor_final = model.evaluate(
    X_test,
    {'out_problema': y_test_prob, 'out_departament': y_test_dep, 'out_urgenta': y_test_urg},
    verbose=0
)

print(f"Loss Total (Eroare): {scor_final[0]:.4f}")
print("--- Acuratețea pe fiecare sarcină ---")
print(f"  Acuratețe Problemă:    {scor_final[4]*100:.2f}%")
print(f"  Acuratețe Departament: {scor_final[5]*100:.2f}%")
print(f"  Acuratețe Urgență:     {scor_final[6]*100:.2f}%")

# --- 8. SALVAREA (Modelul V2) ---
print("\n--- Se salvează modelul și 'traducătoarele' (V2) ---")
model.save('model_dispecer_v2.keras')
joblib.dump(vectorizer, 'vectorizer_v2.joblib')
joblib.dump(encoder_problema, 'encoder_problema_v2.joblib')
joblib.dump(encoder_departament, 'encoder_departament_v2.joblib')
joblib.dump(encoder_urgenta, 'encoder_urgenta_v2.joblib')

print("✅ Modelul Multi-Task și toate componentele au fost salvate!")