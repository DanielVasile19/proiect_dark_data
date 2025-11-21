import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
import numpy as np
import sys

print("-------------------------------------------------")
print("🚀 [Pasul 2 SIMPLU] Antrenare (TF-IDF + MLP Keras)")
print("-------------------------------------------------")

# --- 1. ÎNCĂRCAREA DATELOR ---
NUME_FISIER_CSV = "rapoarte_mentenanta.csv"
try:
    df = pd.read_csv(NUME_FISIER_CSV)
    print(f"✅ Datele sintetice '{NUME_FISIER_CSV}' au fost încărcate. ({len(df)} rânduri)")
except FileNotFoundError:
    print(f"EROARE: Fișierul '{NUME_FISIER_CSV}' nu a fost găsit!")
    sys.exit()

# --- 2. PREGĂTIREA X și y ---
X = df['text_raport'].tolist()
y_text = df['eticheta_problema']

# Transformăm etichetele text (ex. 'motor_defect') în numere (ex. 0)
encoder = LabelEncoder()
y = encoder.fit_transform(y_text)
num_labels = len(np.unique(y))
print(f"✅ Datele X (text) și y (etichete) definite. Avem {num_labels} clase de probleme.")

# Salvăm "dicționarul" de traducere pentru mai târziu
# ex: [motor_defect, pompa_blocata, ...]
numele_claselor = encoder.classes_

# --- 3. ÎMPĂRȚIREA DATELOR ---
X_train_text, X_test_text, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"✅ Datele au fost împărțite: {len(X_train_text)} antrenare, {len(X_test_text)} testare.")

# --- 4. VECTORIZAREA (Noul mod de a transforma textul în numere) ---
print("Se creează 'vocabularul' TF-IDF...")
# max_features=1000 -> ne uităm doar la cele mai frecvente 1000 de cuvinte
vectorizer = TfidfVectorizer(max_features=1000)

# "Învățăm" vocabularul DOAR pe datele de antrenare
X_train_vec = vectorizer.fit_transform(X_train_text)

# "Transformăm" datele de testare folosind același vocabular
X_test_vec = vectorizer.transform(X_test_text)

print(f"✅ Datele text au fost vectorizate. Fiecare text este acum un vector cu {X_train_vec.shape[1]} numere.")

# --- 5. CONSTRUIREA MODELULUI (Keras Simplu) ---
print("Se construiește rețeaua neuronală MLP (Keras)...")

# Aflăm dimensiunea input-ului
dimensiune_input = X_train_vec.shape[1]  # (va fi 1000 sau mai puțin)

# Keras vrea un array dens, nu matricea "sparse" de la TF-IDF
X_train_gata = X_train_vec.toarray()
X_test_gata = X_test_vec.toarray()

model = tf.keras.Sequential([
    # Stratul de intrare (Input Layer)
    tf.keras.layers.InputLayer(input_shape=(dimensiune_input,)),

    # Un strat ascuns (Hidden Layer) cu 64 de neuroni
    # 'relu' este o funcție de activare standard
    tf.keras.layers.Dense(64, activation='relu'),

    # Un strat de Dropout (ajută la prevenirea overfitting-ului)
    tf.keras.layers.Dropout(0.2),

    # Stratul de Ieșire (Output Layer)
    # Are 'num_labels' (ex. 7) neuroni, unul pentru fiecare clasă de problemă
    # 'softmax' transformă ieșirile în probabilități (ex. 90% șansă 'motor_defect')
    tf.keras.layers.Dense(num_labels, activation='softmax')
])

# Compilăm modelul
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',  # Perfect pentru etichetele noastre (0, 1, 2...)
    metrics=['accuracy']
)

# Afișăm arhitectura
model.summary()

# --- 6. ANTRENAREA MODELULUI ---
print("\n--- Începe Antrenarea ---")

history = model.fit(
    X_train_gata,
    y_train,
    epochs=10,  # Putem folosi mai multe epoci, e foarte rapid
    batch_size=16,
    validation_data=(X_test_gata, y_test),
    verbose=1
)

print("✅ Antrenarea este finalizată!")

# --- 7. EVALUAREA FINALĂ ---
print("\n--- Evaluarea Performanței Finale pe Setul de Test ---")
scor_final = model.evaluate(X_test_gata, y_test, verbose=0)

print(f"Loss (Eroare): {scor_final[0]:.4f}")
print(f"Accuracy (Acuratețe): {scor_final[1] * 100:.2f}%")

print("\n-------------------------------------------------")
print("🎉 FELICITĂRI! Ai antrenat un model MLP pe date text!")
print("-------------------------------------------------")