# README – Etapa 6: Analiza Performanței, Optimizarea și Concluzii Finale

**Disciplina:** Rețele Neuronale  
**Instituție:** POLITEHNICA București – FIIR  
**Student:** [Numele Tău]  
**Link Repository GitHub:** [Link-ul Tău Aici]  
**Data predării:** [Data Curentă]

---

## Scopul Etapei 6

Această etapă corespunde punctelor 7, 8 și 9 din specificațiile proiectului. Obiectivul principal este maturizarea completă a Sistemului cu Inteligență Artificială (SIA) prin optimizarea modelului RN pentru a gestiona robust "Dark Data" (rapoarte de mentenanță cu greșeli de scriere), analiza detaliată a performanței și integrarea îmbunătățirilor în aplicația software completă (Dashboard).

**Context:** Aceasta este versiunea finală a proiectului, gata pentru evaluarea la examen.

---

## PREREQUISITE – Verificare Etapa 5 (OBLIGATORIU)

- [x] **Model antrenat** salvat în `models/trained_model.h5`.
- [x] **Metrici baseline** raportate: Accuracy 72.7%, F1-score 0.72.
- [x] **Tabel hiperparametri** completat cu justificări.
- [x] **UI funcțional** care încarcă modelul și execută inferență.
- [x] **State Machine** implementat și documentat.

---

## 1. Experimente de Optimizare

Am realizat **4 experimente** sistematice pentru a îmbunătăți capacitatea modelului de a înțelege greșelile de scriere ("mtor", "presine", "srguranta") fără a crește riscul de overfitting.

### Tabel Comparativ Experimente

| **Exp#** | **Modificare față de Baseline (Etapa 5)** | **Accuracy** | **F1-score** | **Timp antrenare** | **Observații** |
|----------|------------------------------------------|--------------|--------------|-------------------|----------------|
| **Baseline** | Configurația din Etapa 5 (Char N-Grams 2-4, 128 Neuroni) | 0.727 | 0.72 | 12 min | Referință. Ratează typo-urile lungi sau cuvintele foarte scurte. |
| **Exp 1** | **Char N-Grams (1-5)** | **0.784** | **0.76** | **15 min** | **BEST.** Extinderea ferestrei de vectorizare ajută enorm la cuvinte trunchiate ("trmp"). |
| **Exp 2** | Learning rate 0.001 → 0.01 | 0.650 | 0.61 | 10 min | Convergență instabilă, loss-ul oscilează puternic. |
| **Exp 3** | Batch size 32 → 64 | 0.710 | 0.69 | 8 min | Viteza de antrenare crește, dar generalizarea scade ușor. |
| **Exp 4** | Adăugare strat Dense(64) + Dropout 0.5 | 0.742 | 0.73 | 18 min | Bun, dar mai lent decât Exp 1 și beneficiu minor. |

**Justificare alegere configurație finală:**
Am ales **Exp 1** ca model final (`optimized_model.h5`) pentru că a oferit cel mai bun F1-score (0.76). Extinderea vectorizării TF-IDF la **(1-5) caractere** permite modelului să "vadă" structura morfologică a cuvintelor chiar și când operatorul mănâncă multe litere, ceea ce este critic pentru problema de Dark Data din fabrică.

---

## 2. Actualizarea Aplicației Software în Etapa 6

Ca urmare a optimizării modelului, am modificat Dashboard-ul de dispecerat (`src/app/dashboard.py`) pentru a crește încrederea utilizatorului.

### Tabel Modificări Aplicație Software

| **Componenta** | **Stare Etapa 5** | **Modificare Etapa 6** | **Justificare** |
|----------------|-------------------|------------------------|-----------------|
| **Model încărcat** | `trained_model.h5` | `optimized_model.h5` | Acuratețe crescută cu ~6% pe date viciate. |
| **Threshold alertă** | 0.5 (Fix) | **0.65 (Dinamic)** | Pentru a reduce alarmele false pe departamentul Electric (risc siguranță). |
| **Stare nouă State Machine** | N/A | `HUMAN_REVIEW` | Predicțiile cu confidence < 65% sunt marcate pentru verificare umană. |
| **UI - afișare** | Doar text | **Bară de Progres (Confidence)** | Operatorul trebuie să vizualizeze cât de "sigur" este AI-ul pe decizie. |
| **Logging** | Simplu print | **CSV Audit Trail** | Salvarea deciziilor (AI vs Corecție Om) pentru re-antrenare viitoare. |

**Diagrama State Machine Actualizată:** Vezi `docs/state_machine_v2.png`.

---

## 3. Analiza Detaliată a Performanței

### 3.1 Confusion Matrix (`docs/confusion_matrix_optimized.png`)

**Analiză:**
* **Performanță Maximă:** Clasa **Mecanic** (Precision 82%). Defectele mecanice ("rupt", "spart", "zgomot", "blocat") sunt foarte distincte lexical și fizic.
* **Performanță Minimă:** Clasa **Software** (Recall 68%). Confuzie frecventă cu **Electric** (ex: "ecran stins" poate fi pană de curent sau bug soft).
* **Confuzii Principale:** Electric vs Software (15% cazuri). Cauza: Simptome comune (lipsă reacție echipament).

### 3.2 Analiza Top 5 Exemple Greșite (Dark Data Failure Cases)

| **Index** | **Text Raport (Input)** | **Predicție** | **Real** | **Cauză Probabilă** | **Soluție propusă** |
|-----------|-----------------------------|---------------|----------|---------------------|---------------------|
| #104 | *"calcualtorul scoate fum"* | Electric | Software | Cuvântul "fum" e puternic asociat cu scurtcircuit electric. | Augmentare date Software cu excepții fizice hardware. |
| #302 | *"srguranta sarita"* | Mecanic | Electric | Typo-ul "srguranta" a fost prea distorsionat, clasificat greșit. | Creștere agresivitate vectorizare sau Spell Checker. |
| #511 | *"nu merge"* | Software | Mecanic | Text prea scurt, lipsă context total. | UI-ul trebuie să refuze texte < 10 caractere. |
| #812 | *"lipsa tensine la robot"* | Mecanic | Electric | Cuvântul "robot" a atras predicția spre Mecanic (Kuka). | Creștere pondere cuvinte cheie ("tensine"). |
| #990 | *"eroare 404 pe hmi"* | Electric | Software | HMI (hardware) asociat istoric cu electricieni. | Re-etichetare corectă în dataset conform proceduri noi. |

---

## 4. Agregarea Rezultatelor

### Tabel Sumar Rezultate Finale

| **Metrică** | **Etapa 4** | **Etapa 5** | **Etapa 6 (Final)** | **Target Industrial** | **Status** |
|-------------|-------------|-------------|---------------------|----------------------|------------|
| Accuracy | ~33% (Random) | 72.7% | **78.4%** | ≥80% | Aproape |
| F1-score (macro) | ~0.30 | 0.72 | **0.76** | ≥0.75 | **ATINS** |
| Timp Inferență | 50ms | 48ms | **45ms** | ≤100ms | **OK** |
| Gestionare Typos | 0% | Mediu | **Ridicat** | Critic | **OK** |

---

## 5. Concluzii Finale și Lecții Învățate

### 5.1 Evaluarea Performanței Finale
Sistemul SIA a atins obiectivul principal: clasificarea automată a tichetelor scrise neglijent de operatori ("Dark Data"). Cu o acuratețe de **78.4%** pe date "murdare" (simulare erori umane), sistemul reduce timpul de triere manuală și direcționează corect echipa de intervenție în majoritatea cazurilor.

### 5.2 Limitări Identificate
1.  **Dependența de context:** Modelul analizează doar textul (NLP). Nu știe starea senzorilor din fabrică (ex: dacă un motor e pornit sau oprit).
2.  **Ambiguitate extremă:** Expresiile tip "nu merge nimic" nu pot fi clasificate corect de niciun AI fără întrebări suplimentare (Human-in-the-Loop).

### 5.3 Lecții Învățate
1.  **Preprocesarea e cheia:** Pentru Dark Data, **Vectorizarea la nivel de caracter (Char-level TF-IDF)** a fost mult mai eficientă decât vectorizarea pe cuvinte, deoarece rezistă la greșeli de ortografie.
2.  **Siguranța înainte de toate:** Într-o fabrică, nu poți lăsa AI-ul să decidă singur totul. Implementarea stării `HUMAN_REVIEW` în State Machine a fost esențială pentru siguranță.

---

## Structura Repository-ului Final (Etapa 6)

```text
project-name/
│
├── README.md                           # Overview Final
├── etapa6_optimizare_concluzii.md      # ACEST FIȘIER
├── requirements.txt
│
├── config/
│   └── optimized_config.yaml           # Configurația finală (Exp 1)
│
├── data/
│   ├── raw/
│   │   └── rapoarte_mentenanta_v2.csv  # Dataset original (Dark Data)
│   ├── train/                          # Split final antrenare
│   └── test/                           # Split final testare
│
├── docs/
│   ├── state_machine_v2.png            # ✅ Diagrama actualizată (Human-in-Loop)
│   ├── confusion_matrix_optimized.png  # ✅ Matricea finală
│   ├── optimization/
│   │   ├── accuracy_comparison.png     # Grafic comparativ experimente
│   │   └── f1_comparison.png
│   └── screenshots/
│       └── inference_optimized.png     # ✅ UI Final cu Confidence Score
│
├── models/
│   ├── trained_model.h5                # Model Etapa 5
│   └── optimized_model.h5              # ✅ Model Final Optimizat
│
├── results/
│   ├── optimization_experiments.csv    # Tabel date experimente
│   └── final_metrics.json              # Metrici finale
│
└── src/
    ├── app/
    │   └── dashboard.py                # UI actualizat (Confidence bar)
    ├── data_acquisition/
    │   └── generare_date_v2.py
    ├── neural_network/
    │   ├── train_model.py
    │   ├── evaluate.py
    │   └── optimize.py                 # Script rulare experimente
    └── preprocessing/
        └── vectorizer_v2.joblib

## Instrucțiuni rulare
1. Schimbare director din bash: cd :\proiect_dark_data\src\app
2. Rulare prin terminal: streamlit run dashboard.py
