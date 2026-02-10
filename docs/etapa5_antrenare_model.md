# 📘 README – Etapa 5: Configurarea și Antrenarea Modelului RN

**Disciplina:** Rețele Neuronale  
**Instituție:** POLITEHNICA București – FIIR  
**Student:** [Numele Tău]  
**Data predării:** [Data Curentă]

---

## Scopul Etapei 5

Antrenarea efectivă a modelului Multi-Task definit în Etapa 4, capabil să clasifice rapoarte de mentenanță scrise greșit (Dark Data). Obiectivul este obținerea unei acurateți de bază (Baseline) și integrarea modelului antrenat în Dashboard-ul de dispecerat.

---

## PREREQUISITE – Verificare Etapa 4

- [x] **State Machine** definit în `docs/state_machine.png`.
- [x] **Contribuție 100% date originale** (generate prin scriptul `generare_date_v2.py` cu simulare erori umane).
- [x] **Modul 1 (Data Logging)** funcțional.
- [x] **Modul 2 (RN)** definit (Multi-Output Model).
- [x] **Modul 3 (UI)** funcțional (Streamlit).

---

## 1. Pregătire Date și Antrenare (Nivel 1)

### Configurația Antrenării

Am antrenat modelul pe setul de date generat (12.000 observații), folosind o arhitectură care prelucrează textul la nivel de caracter (pentru robustețe la typos).

* **Metrici obținute (Test Set):**
    * **Acuratețe (Medie pe 3 task-uri):** 72.7%
    * **F1-Score (Macro):** 0.72
* **Model salvat:** `models/trained_model.h5`

### Tabel Hiperparametri și Justificări

| **Hiperparametru** | **Valoare Aleasă** | **Justificare** |
|--------------------|-------------------|-----------------|
| **Batch size** | 32 | Dataset-ul are 12k exemple. 32 oferă un balans bun între viteza de actualizare a ponderilor și stabilitatea gradientului. |
| **Number of epochs** | 15 | Cu **Early Stopping** (patience=3). Modelul converge rapid datorită naturii repetitive a limbajului tehnic, chiar și cu erori. |
| **Optimizer** | Adam (lr=0.001) | Standardul industriei pentru NLP. Gestionează bine vectorii rari (sparse vectors) generați de TF-IDF. |
| **Loss Function** | Sparse Categorical Crossentropy | Avem clasificare multi-class pentru fiecare ieșire (Departament, Problemă, Urgență), iar etichetele sunt integers (LabelEncoded). |
| **Vectorizare** | Char-WB (2-4 n-grams) | **Critic pentru proiect:** Nu analizăm cuvinte, ci secvențe de caractere. Asta permite modelului să învețe că "mtor" ≈ "motor". |
| **Neurons (Hidden)** | 128 (Dense) | Suficient pentru a captura corelațiile dintre n-gramele de caractere și tipul defecțiunii. |

---

## 2. Analiză Erori în Context Industrial (Nivel 2)

### 1. Pe ce clase greșește cel mai mult modelul?
Modelul face confuzii între **Mecanic** și **Electric** la simptomele ambigue descrise scurt.
* Exemplu: "miroase a ars la banda 2".
* Modelul prezice uneori *Mecanic* (frecare curea), alteori *Electric* (motor ars). Fără detalii suplimentare ("fum", "flamă"), ambiguitatea este reală.

### 2. Ce caracteristici ale datelor cauzează erori?
**"Dark Data" extrem:** Când greșelile de tastare distrug mai mult de 50% din cuvânt.
* Exemplu: "srz ars" (în loc de "senzor ars").
* N-gramele rezultate (`sr`, `rz`) sunt prea comune și nu se leagă semantic puternic de clasa "Senzori".

### 3. Ce implicații are pentru aplicația industrială?
* **False Negatives pe Urgență:** Dacă o problemă "Critica" (ex: cablu sectionat) este clasificată ca "Medie", există risc de electrocutare.
* **Soluție:** Am implementat în UI un mecanism *Human-in-the-Loop*. Dacă "Confidence Score" < 60%, sistemul avertizează dispecerul să verifice manual.

### 4. Ce măsuri corective propuneți?
1.  **Augmentare date:** Generarea mai multor variații de prescurtări agresive (ex: "pt" pentru "pentru", "temp" pentru "temperatura").
2.  **Thresholding:** Setarea unui prag minim de siguranță (0.6) sub care tichetul este marcat automat pentru revizie umană.

---

## 3. Structura Repository-ului la Finalul Etapei 5
```
project-name/
├── README_Etapa5_Antrenare_RN.md
├── requirements.txt
├── data/
│   ├── raw/rapoarte_mentenanta_v2.csv
│   ├── train/
│   └── test/
├── docs/
│   ├── screenshots/inference_real.png  # ✅ Screenshot nou
│   └── loss_curve.png
├── models/
│   ├── trained_model.h5                # ✅ Model antrenat
│   ├── encoder_departament_v2.joblib
│   └── encoder_problema_v2.joblib
├── results/training_history.csv        # Log antrenare
└── src/
    ├── app/dashboard.py
    ├── neural_network/train_model.py
    └── preprocessing/vectorizer_v2.joblib
```
## 4. Instrucțiuni de Rulare

1.  **Antrenare Model:**
    ```bash
    python src/neural_network/train_model.py
    ```
    *Va genera fișierul `models/trained_model.h5` și graficele în `docs/`.*

2.  **Lansare Dashboard:**
    ```bash
    streamlit run src/app/dashboard.py
    ```
    *Acum puteți testa cu texte reale (ex: "motorul s-a oprit brusc") și veți primi predicții bazate pe rețeaua antrenată.*