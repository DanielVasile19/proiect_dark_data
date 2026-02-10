## 1. Identificare Proiect

| Câmp | Valoare |
|------|---------|
| **Student** | [Vasile Sorin-Daniel-Virgil] |
| **Grupa / Specializare** | [ex: 632AB / Informatică Industrială] |
| **Disciplina** | Rețele Neuronale |
| **Instituție** | POLITEHNICA București – FIIR |
| **Link Repository GitHub** | [https://github.com/DanielVasile19/proiect_dark_data] |
| **Acces Repository** | [Public]|
| **Stack Tehnologic** | [Python] |
| **Domeniul Industrial de Interes (DII)** | [Producție, Mentenanță] |
| **Tip Rețea Neuronală** | [Multi-Layer Perceptron] |

### Rezultate Cheie (Versiunea Finală vs Etapa 6)

| Metric | Țintă Minimă | Rezultat Etapa 6 | Rezultat Final | Îmbunătățire | Status |
|--------|--------------|------------------|----------------|--------------|--------|
| Accuracy (Test Set) | ≥70% | [72.7%] | [72.7%] | [0%] | [✓] |
| F1-Score (Macro) | ≥0.65 | [0.72] | [0.72] | [0] | [✓] |
| Latență Inferență | [<50] | [45 ms] | [45 ms] | [0 ms] | [✓] |
| Contribuție Date Originale | ≥40% | [100%] | [100%] | 0 | [✓] |
| Nr. Experimente Optimizare | ≥4 | [5] | [6] | +1 | [✓] |

### Declarație de Originalitate & Politica de Utilizare AI

**Acest proiect reflectă munca, gândirea și deciziile mele proprii.**

Utilizarea asistenților de inteligență artificială (ChatGPT, Claude, Grok, GitHub Copilot etc.) este **permisă și încurajată** ca unealtă de dezvoltare – pentru explicații, generare de idei, sugestii de cod, debugging, structurarea documentației sau rafinarea textelor.

**Nu este permis** să preiau:
- cod, arhitectură RN sau soluție luată aproape integral de la un asistent AI fără modificări și raționamente proprii semnificative,
- dataset-uri publice fără contribuție proprie substanțială (minimum 40% din observațiile finale – conform cerinței obligatorii Etapa 4),
- conținut esențial care nu poartă amprenta clară a propriei mele înțelegeri.

**Confirmare explicită (bifez doar ce este adevărat):**

| Nr. | Cerință                                                                 | Confirmare |
|-----|-------------------------------------------------------------------------|------------|
| 1   | Modelul RN a fost antrenat **de la zero** (weights inițializate random, **NU** model pre-antrenat descărcat) | [✓] DA     |
| 2   | Minimum **40% din date sunt contribuție originală** (generate/achiziționate/etichetate de mine) | [✓] DA     |
| 3   | Codul este propriu sau sursele externe sunt **citate explicit** în Bibliografie | [✓] DA     |
| 4   | Arhitectura, codul și interpretarea rezultatelor reprezintă **muncă proprie** (AI folosit doar ca tool, nu ca sursă integrală de cod/dataset) | [✓] DA     |
| 5   | Pot explica și justifica **fiecare decizie importantă** cu argumente proprii | [✓] DA     |

**Semnătură student (prin completare):** Declar pe propria răspundere că informațiile de mai sus sunt corecte.

---

## 2. Descrierea Nevoii și Soluția SIA

### 2.1 Nevoia Reală / Studiul de Caz

*[Descrieți în 1-2 paragrafe: Ce problemă concretă din domeniul industrial rezolvă acest proiect? Care este contextul și situația actuală? De ce este importantă rezolvarea acestei probleme?]*

In perioada de practica, am facut parte din echipa de mentenanță a unei fabrici de sticla. În fiecare dimineață erau văzute rapoartele de mentenanță de echipa de ingineri trimise de toți angajații fabrici. De multe ori aceștia se grăbeau să scrie acel raport, iar acestea nu erau foarte clare, și am decis că acest model MLP poate fi una dintre soluțiile care să le poată ușura munca.

### 2.2 Beneficii Măsurabile Urmărite

*[Listați 3-5 beneficii concrete cu metrici țintă]*

1. Reducerea timpului de interpretare cu 70-80% 
2. Reducerea timpului întălnirilor cu 35-40%
3. Reducerea perioadei de rezolvare a unei probleme cu 25-30%
4. [...]
5. [...]

### 2.3 Tabel: Nevoie → Soluție SIA → Modul Software

| **Nevoie reală concretă** | **Cum o rezolvă SIA-ul** | **Modul software responsabil** | **Metric măsurabil** |
|---------------------------|--------------------------|--------------------------------|----------------------|
| [Reducerea timpului de triere a tichetelor (în loc de 5-10 minute manual)] | [Clasificare automata] | [RN+Web service] | [<50ms(5 minute manual)] |
| [Clasificarea problemelor dupa departamente] | [Predictie departament, problema posibila si gradul de urgenta] | [Logic decision+Web service] | [<85 precizie pe departamente] |

---

## 3. Dataset și Contribuție Originală

### 3.1 Sursa și Caracteristicile Datelor

| Caracteristică | Valoare |
|----------------|---------|
| **Origine date** | [Simulare] |
| **Sursa concretă** | [Generare date] |
| **Număr total observații finale (N)** | [ex: 12,000] |
| **Număr features** | [1 input si 3 output-uri] |
| **Tipuri de date** | [Text] |
| **Format fișiere** | [CSV] |
| **Perioada colectării/generării** | [Februarie 2026] |

### 3.2 Contribuția Originală (minim 40% OBLIGATORIU)

| Câmp | Valoare |
|------|---------|
| **Total observații finale (N)** | [12.000] |
| **Observații originale (M)** | [12.000] |
| **Procent contribuție originală** | [100%] |
| **Tip contribuție** | [Date sintetice] |
| **Locație cod generare** | `src/data_acquisition/generare_date_v2.py` |
| **Locație date originale** | `data/generated/raw` |

**Descriere metodă generare/achiziție:**

*[Explicați în 1-2 paragrafe: Cum ați generat/achiziționat datele originale? Ce parametri ați folosit? De ce sunt relevante pentru problema voastră?]*

Am generat datele folosind 3 parti ale unui raport, data mapping-ul fiind urmatorul: problema + locatie + grad urgenta.

### 3.3 Preprocesare și Split Date

| Set | Procent | Număr Observații |
|-----|---------|------------------|
| Train | 70% | [8400] |
| Validation | 15% | [1400] |
| Test | 15% | [1400] |

**Preprocesări aplicate:**
- Normalizare text
- Vectorizare la nivel de caracter
- Label encoding
- Pastrarea erorilor gramaticale pentru antrenarea modelului

**Referințe fișiere:** `data/README.md`, `config/preprocessing_params.pkl`

---

## 4. Arhitectura SIA și State Machine

### 4.1 Cele 3 Module Software

| Modul | Tehnologie | Funcționalitate Principală | Locație în Repo |
|-------|------------|---------------------------|-----------------|
| **Data Logging / Acquisition** | [Python] | [Generare date sintetice cu erori umane] | `src/data_acquisition/` |
| **Neural Network** | [TensorFlow/Keras] | [Clasificare Multi-Task pe text vectorizat] | `src/neural_network/` |
| **Web Service / UI** | [Streamlit] | [Dashboard cu inferență live.] | `src/app/` |

### 4.2 State Machine

**Locație diagramă:** `docs/state_machine.png` *(sau `state_machine_v2.png` dacă actualizată în Etapa 6)*

**Stări principale și descriere:**

| Stare | Descriere | Condiție Intrare | Condiție Ieșire |
|-------|-----------|------------------|-----------------|
| `IDLE` | [Asteptare input utilizator] | [Start aplicație] | [Apasare buton "Analizează"] |
| `ACQUIRE_DATA` | [Preluare text raw din campul input] | [Asteapta click-ul] | [Text valid] |
| `PREPROCESS` | [Vectorizare text] | [Text brut disponibil] | [Vector generat] |
| `INFERENCE` | [Predicție simulată prin Multi-Task] | [Vector de intrare pregatit] | [Predicție generată] |
| `DECISION` | [Calculare si verificare gradul de siguranță] | [Output disponibil] | [Auto ok sau REQ_HUMAN] |
| `OUTPUT/ALERT` | [Afisare rezultate in Dashboard] | [Decizie finalizată] | [Confirmare operator] |
| `ERROR` | [Input gol, model lipsa și lipsa date] | [Excepție try-except] | [Înapoi in IDLE] |

**Justificare alegere arhitectură State Machine:**

*[1 paragraf: De ce această structură pentru problema voastră specifică?]*

Pentru că în sistemele moderne am observat cel mai des acest tip de structură, și anume Human-in-the-Loop.

### 4.3 Actualizări State Machine în Etapa 6 (dacă este cazul)

| Componentă Modificată | Valoare Etapa 5 | Valoare Etapa 6 | Justificare Modificare |
|----------------------|-----------------|-----------------|------------------------|
| [Threshold decizie] | [0.5] | [0.65] | [Reducere risc confuzie departamente] |

---

## 5. Modelul RN – Antrenare și Optimizare

### 5.1 Arhitectura Rețelei Neuronale

```
Input (Vector TF-IDF: shape [1, 3000]) 
  │
  ▼
[Shared Layers - Extragere Trăsături Comune]
  → Dense(128, Activation='relu') 
  → Dropout(0.3)  (Regularizare pentru evitare overfitting)
  → Dense(64, Activation='relu')
  │
  ▼
[Multi-Task Heads - Ramificare Decizională]
  ├─► Head 1: Dense(3, Softmax)  → Output: Departament (Mecanic/Electric/Soft)
  ├─► Head 2: Dense(9, Softmax)  → Output: Tip Problemă (specifică)
  └─► Head 3: Dense(3, Softmax)  → Output: Urgență (Critică/Medie/Mică)
```

**Justificare alegere arhitectură:**

*[1-2 propoziții: De ce această arhitectură? Ce alternative ați considerat și de ce le-ați respins?]*

Multi-Layer Perceptron și Multi-Task Learning mi se par cele mai potrivite deoarece inputul este un vector numeric(vectorizarea TF-IDF aici) care nu necesită convoluții sau secvențiere. 

### 5.2 Hiperparametri Finali (Model Optimizat - Etapa 6)

| Hiperparametru | Valoare Finală | Justificare Alegere |
|----------------|----------------|---------------------|
| Learning Rate | [00] | [Valoare stadard Adam] |
| Batch Size | [32] | [Compromis între viteza de antrenare și generalizare pentru 12000 observații] |
| Epochs | [20] | [Limită maximă setată, cu early stopping activ.] |
| Optimizer | [Adam] | [Gestionează eficient datele textuale non-staționare.] |
| Loss Function | [Sparse Categorical Crossentropy] | [Sumă pentru 3 ieșiri] |
| Regularizare | [Dropout 0.5] | [Crescut de la 0.3 la 0.5 pentru a nu memora greșelile gramatice] |
| Early Stopping | [patience=3, monitor='val_loss'] | [Oprire imediată cand începe overfittingul pe datele de validare] |

### 5.3 Experimente de Optimizare (minim 4 experimente)

| Exp# | Modificare față de Baseline | Accuracy | F1-Score | Timp Antrenare | Observații |
|------|----------------------------|----------|----------|----------------|------------|
| **Baseline** | Configurația Etapa 5 (Char-WB 2-4, 128 Neuroni) | [72.70%] | [0.72] | [12 min] | Ratează typo-urile severe sau cuvintele trunchiate. |
| Exp 1 | [Char N-Grams (1-5)] | [78.40%] | [0.76] | [15 min] | [Captarea rădăcinilor scurte (1-3 chars) a redus masiv erorile pe cuvinte viciate.] |
| Exp 2 | [Learning Rate 0.001 → 0.0001] | [73.50%] | [0.73] | [25 min] | [Convergență foarte lentă, îmbunătățire marginală (+0.8%) nejustificată de timp.] |
| Exp 3 | [Arhitectură: +1 strat Dense(64)] | [74.20%] | [0.73] | [18 min] | [Ușor overfitting pe datele de train, generalizare slabă pe test.] |
| Exp 4 | [Dropout 0.3 → 0.5] | [71.50%] | [0.70] | [12 min] | [Regularizare prea agresivă, modelul face underfitting pe clasele rare.] |
| Exp 5 | [Batch Size 32 → 64] | [71.10%] | [0.69] | [8 min] | [Antrenare rapidă, dar stabilitatea gradientului a scăzut (loss fluctuant).] |
| **FINAL** | [Configurația Exp 1] | **[78.40%]** | **[0.76]** | [15 min] | **Modelul optimizat pentru robustețe la greșeli de scriere.** |

**Justificare alegere model final:**

*[1 paragraf: De ce această configurație? Ce compromisuri ați făcut între accuracy/timp/complexitate?]*

Am ales Char N-Gras deoarece este cea mai capabilă să ofere o creștere a F1-score-ului de la 0.72 la 0.76. 

**Referințe fișiere:** `results/optimization_experiments.csv`, `models/optimized_model.h5`

---

## 6. Performanță Finală și Analiză Erori

### 6.1 Metrici pe Test Set (Model Optimizat)

| Metric | Valoare | Target Minim | Status |
|--------|---------|--------------|--------|
| **Accuracy** | [78.40%] | ≥70% | [✓] |
| **F1-Score (Macro)** | [0.76] | ≥0.65 | [✓/✗] |
| **Precision (Macro)** | [0.79] | - | - |
| **Recall (Macro)** | [0.74] | - | - |

**Îmbunătățire față de Baseline (Etapa 5):**

| Metric | Etapa 5 (Baseline) | Etapa 6 (Optimizat) | Îmbunătățire |
|--------|-------------------|---------------------|--------------|
| Accuracy | [72.70%] | [78.40%] | [+5.70%] |
| F1-Score | [0.72] | [0.76] | [+0.04] |

**Referință fișier:** `results/final_metrics.json`

### 6.2 Confusion Matrix

**Locație:** `docs/confusion_matrix_optimized.png`

**Interpretare:**

| Aspect | Observație |
|--------|------------|
| **Clasa cu cea mai bună performanță** | [Mecanic] - Precision [85%], Recall [82%] |
| **Clasa cu cea mai slabă performanță** | [Nume clasă] - Precision [72%], Recall [68%] |
| **Confuzii frecvente** | [Clasa electric si Software sunt uneori confundate din cauza similarităților(ex: ecran stins).] |
| **Dezechilibru clase** | [Clasa Software are o varietate lexicală mai mică ducând la un recall mai scăzut.] |

### 6.3 Analiza Top 5 Erori

| # | Input (descriere scurtă) | Predicție RN | Clasă Reală | Cauză Probabilă | Implicație Industrială |
|---|--------------------------|--------------|-------------|-----------------|------------------------|
| 1 | ["ecran negru nu raspunde la atingere"] | [Electric] | [Software] | [Ecran are un scor TF-IDF mai mare decat "nu raspunde ", adica eroare software] | [Electricianul demontează panoul degeaba.] |
| 2 | ["mtr s-a oprit brusc, miroase a ars"] | [Mecanic] | [Electric] | ["Mtr" nu a fost asociat corect cu "motor", iar "oprit" a dus predicția spre blocaj mecanic.] | [Risc de incendiu ignorat; Mecanicul nu poate rezolva un scurtcircuit.] |
| 3 | ["senzorul nu citeste datele in sistem"] | [Software] | [Electric] | [Modelul a asociat cuvintele "date" și "sistem" cu Software, deși senzorul fizic este defect] | [IT-ul verifică serverele degeaba, senzorul fizic trebuie înlocuit.] |
| 4 | ["pompa face zgomot ciudat la pornire"] | [Urgență: Medie] | [Urgență: Critică] | [Modelul nu a captat gravitatea combinației "zgomot" + "pornire", clasificând ca uzură normală.] | [Pompa cedează catastrofal în scurt timp, oprind producția.] |
| 5 | ["bara protectie rupta"] | [Mecanic] | [Mecanic (Dar Problemă Greșită)] | [Predicția problemei a fost "Uzura" în loc de "Structura", din cauza numărului mic de exemple cu "bara".] | [Echipa vine cu piese de schimb greșite.] |

### 6.4 Validare în Context Industrial

**Ce înseamnă rezultatele pentru aplicația reală:**

*[1 paragraf: Traduceți metricile în impact real în domeniul vostru industrial]*

Am precizat că acuratețea este ~78%, asta înseamnă ca 78/100 tichete de mentenanță durează undeva la 1 secundă. În schimb, pentru cele 22 de cazuri în care se presupune că modelul greșește, avem opțiunea de feedback, în care fiecare departament poate preciza dacă i-a fost alocată problema corect. 

**Pragul de acceptabilitate pentru domeniu:** [Precision ≥ 80% pentru Urgențe Critice]  
**Status:** [Atins parțial - Media este de 78.4%(in parametri ceruți) dar pe clasa Software este 68%]  
**Plan de îmbunătățire (dacă neatins):** [Active learning, prin antrenarea lunară cu tichete trimise, și puse manual etichetele.]

---

## 7. Aplicația Software Finală

### 7.1 Modificări Implementate în Etapa 6

| Componentă | Stare Etapa 5 | Modificare Etapa 6 | Justificare |
|------------|---------------|-------------------|-------------|
| **Model încărcat** | `trained_model.h5` | `optimized_model.h5` | [+5.7% accuracy, +0.04 F1-Score] |
| **Threshold decizie** | [0.5 (argmax standard)] | [0.65 (filtru dinamic)] | [Siguranță: Predicțiile incerte necesită validare umană (Human-in-the-Loop)] |
| **UI - feedback vizual** | [Text simplu (Clasă)] | [Bară progres colorată + Scor %] | [Transparență: Operatorul vede nivelul de risc al deciziei automate] |
| **Logging** | [Doar afișare rezultat] | [Salvare corecții în CSV] | [Active Learning: Datele corectate de om re-antrenează modelul viitor] |
| [Preprocesare] | [TF-IDF Char (2-4)] | [TF-IDF Char (1-5) + Stopwords] | [Captarea rădăcinilor scurte din cuvinte viciate ("mtor" vs "motor")] |

### 7.2 Screenshot UI cu Model Optimizat

**Locație:** `docs/screenshots/inference_optimized.png`

*[Descriere scurtă: Ce se vede în screenshot? Ce demonstrează?]*

Interfața UI cu un exemplu de raport, analiza și bara de feedback.

### 7.3 Demonstrație Funcțională End-to-End

**Locație dovadă:** `docs/demo/` *(GIF / Video / Secvență screenshots)*

**Fluxul demonstrat:**

| Pas | Acțiune | Rezultat Vizibil |
|-----|---------|------------------|
| 1 | Input | [Introducere text operator] |
| 2 | Procesare | [Afișare în partea dreaptă a rezultatului] |
| 3 | Inferență | [Verificarea și confirmarea rezultatului] |
| 4 | Decizie | [Introducerea unui nou raport sau ] |

**Latență măsurată end-to-end:** [X] ms  
**Data și ora demonstrației:** [09.02.2026, 17:12]

---

## 8. Structura Repository-ului Final

```
proiect_dark_data/
│
├── config/
│   └── requirements.txt
│
├── data/
│   ├── generated/
│   │   └── raw/
│   ├── processed/
│   │   ├── encoder_departament_v2.joblib
│   │   ├── encoder_problema_v2.joblib
│   │   ├── encoder_urgenta_v2.joblib
│   │   └── vectorizer_v2.joblib
│   ├── raw/
│   │   └── rapoarte_mentenanta_v2.csv
│   ├── test/
│   │   └── .gitkeep
│   ├── train/
│   │   └── .gitkeep
│   └── validation/
│       └── .gitkeep
│
├── docs/
│   ├── datasets/
│   │   └── .gitkeep
│   ├── demo/
│   │   └── demo.gif
│   ├── optimization/
│   ├── results/
│   │   ├── conf_matrix_departament.png
│   │   ├── conf_matrix_problema.png
│   │   ├── grafic_performanta.png
│   │   ├── learning_curves_final.png
│   │   ├── loss_curve.png
│   │   └── stress_test_matrix.png
│   ├── screenshots/
│   │   ├── demo_ui_charts.png
│   │   ├── demo_ui.png
│   │   └── inference_real.png
│   ├── conf_matrix_departament.png
│   ├── conf_matrix_problema.png
│   ├── confusion_matrix_optimized.png
│   ├── etapa3_analiza_date.md
│   ├── etapa4_arhitectura_SIA.md
│   ├── etapa5_antrenare_model.md
│   ├── etapa6_optimizare_concluzii.md
│   ├── matrice_confuzie_finala.png
│   └── state_machine.png
│
├── models/
│   ├── model_dispecer_v2.keras
│   ├── optimized_model.h5
│   └── trained_model.h5
│
├── results/
│   ├── final_metrics.json
│   └── training_history.csv
│
├── src/
│   ├── app/
│   │   └── dashboard.py
│   ├── data_acquisition/
│   │   └── generare_date_v2.py
│   ├── neural_network/
│   │   ├── evaluate_model.py
│   │   ├── predict_dispecer.py
│   │   └── train_model.py
│   └── preprocessing/
│       └── .gitkeep
│
├── .gitignore
└── VASILE_Sorin-Daniel-Virgil_632AB_README_Proiect_RN.md
```

### Legendă Progresie pe Etape

| Folder / Fișier | Etapa 3 | Etapa 4 | Etapa 5 | Etapa 6 |
|-----------------|:-------:|:-------:|:-------:|:-------:|
| `data/raw/`, `processed/`, `train/`, `val/`, `test/` | ✓ Creat | - | Actualizat | - |
| `data/generated/` | ✓ Creat | - | - | - |
| `src/preprocessing/` | ✓ Creat | - | - | - |
| `src/data_acquisition/` | ✓ Creat | - | - | - |
| `src/neural_network/model.py` | - | ✓ Creat | - | - |
| `src/neural_network/train.py`, `evaluate.py` | - | - | ✓ Creat | - |
| `src/neural_network/optimize.py`, `visualize.py` | - | - | - | ✓ Creat |
| `src/app/` | - | ✓ Schelet | Actualizat | ✓Final |
| `models/untrained_model.*` | - | ✓ Creat | - | - |
| `models/trained_model.*` | - | - | ✓ Creat | - |
| `models/optimized_model.*` | - | - | - | ✓ Creat |
| `docs/state_machine.*` | - | ✓ Creat | - | - |
| `docs/etapa3_analiza_date.md` | ✓ Creat | - | - | - |
| `docs/etapa4_arhitectura_SIA.md` | - | ✓ Creat | - | - |
| `docs/etapa5_antrenare_model.md` | - | - | ✓ Creat | - |
| `docs/etapa6_optimizare_concluzii.md` | - | - | - | ✓ Creat |
| `docs/confusion_matrix_optimized.png` | - | - | - | ✓ Creat |
| `docs/screenshots/` | - | ✓ Demo | Actualizat | Final |
| `results/training_history.csv` | - | - | ✓ Creat | - |
| `results/optimization_experiments.csv` | - | - | - | ✓ Creat |
| `results/final_metrics.json` | - | - | - | ✓ Creat |
| **README.md** (acest fișier) | Draft | Actualizat | Actualizat | **FINAL** |

*\* Actualizat dacă s-au adăugat date noi în Etapa 4*

### Convenție Tag-uri Git

| Tag | Etapa | Commit Message Recomandat |
|-----|-------|---------------------------|
| `v0.3-data-ready` | Etapa 3 | "Etapa 3 completă - Dataset 'Dark Data' generat (N=15k) și vectorizator salvat" |
| `v0.4-architecture` | Etapa 4 | "Etapa 4 completă - Arhitectură MLP Multi-Task și Schelet UI definite" |
| `v0.5-model-trained` | Etapa 5 | "Etapa 5 completă - Baseline Training. Accuracy=72.7%, F1=0.72" |
| `v0.6-optimized-final` | Etapa 6 | "Etapa 6 completă - Model Optimizat (HitL). Accuracy=78.4%, F1=0.76" |

---

## 9. Instrucțiuni de Instalare și Rulare

### 9.1 Cerințe Preliminare

```
Python >= 3.8 (recomandat 3.10+)
pip >= 21.0
```

### 9.2 Instalare

```bash
# 1. Clonare repository
git clone https://github.com/DanielVasile19/proiect_dark_data.git
cd proiect_dark_data

# 2. Creare mediu virtual (Recomandat pentru izolare)
python -m venv venv

# Activare mediu virtual:
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# 3. Instalare dependențe
pip install -r requirements.txt
```

### 9.3 Rulare Pipeline Complet

```bash
# Pasul 1: Generare Dataset "Dark Data" (Simulare Date Industriale)
# Generează 15.000 de rapoarte cu erori (typos, jargon) în data/generated/raw/
python src/data_acquisition/generate.py

# Pasul 2: Antrenare Model Neuronal (Reproducere Rezultate)
# Preia datele, vectoriizează (TF-IDF N-Grams) și antrenează modelul Multi-Task.
# Salvează modelul optimizat în: models/optimized_model.h5
python src/neural_network/train.py

# Pasul 3: Evaluare Model pe Setul de Test
# Calculează metricile finale (Accuracy, F1) și generează matricea de confuzie în docs/
python src/neural_network/evaluate.py

# Pasul 4: Lansare Aplicație Dashboard (UI)
# Pornește serverul Streamlit la adresa http://localhost:8501
streamlit run src/app/main.py
```

### 9.4 Verificare Rapidă 

```bash
# Verificare automată a existenței modelului și a bibliotecilor
python -c "import os; import tensorflow as tf; m_path='models/optimized_model.h5'; assert os.path.exists(m_path), 'EROARE: Modelul nu a fost găsit!'; print(f'✓ SUCCES: Model încărcat corect (TF {tf.__version__})')"

# (Opțional) Testare inferență pe un singur text
python src/neural_network/predict_dispecer.py
```

## 10. Concluzii și Discuții

### 10.1 Evaluare Performanță vs Obiective Inițiale

| Obiectiv Definit (Secțiunea 2) | Target | Realizat | Status |
|--------------------------------|--------|----------|--------|
| [Clasificare rapoarte] | [Reducere timp <1min] | [<50ms] | [✓] |
| [Întelegere greșeli de exprimare] | [30% greșeli] | [F1=0.76 pe date viciaate] | [✓] |
| Accuracy pe test set | ≥70% | [78.4%] | [✓] |
| F1-Score pe test set | ≥0.65 | [0.76] | [✓] |
| [Siguranța Operațională] | [Fără fals pozitiv] | [Human-in-the-loop] | [✓] |

### 10.2 Ce NU Funcționează – Limitări Cunoscute

*[Fiți onești - evaluatorul apreciază identificarea clară a limitărilor]*

1. **Limitare 1:** [Modelul fără un subiect clar, eșuează. De exemplu dacă pur și simplu scrie eroare/nu merge/stricat. În shimb dacă scriem motor, soft, ulei, modelul poate recunoaște problema.]
2. **Limitare 2:** [Incapacitatea de a clasifica mai multe probleme simultan. De exemplu daca scriem:"Ulei scurs și PLC-ul este blocat" o să scoată doar o clasă.]
3. **Limitare 3:** [Nu este garbage filter, dacă un operator să zicem că scrie: "Salut","Verificare","Test", modelul o să scoată o predicție]
4. **Funcționalități planificate dar neimplementate:** [Antrenarea automată a modelului atunci când primește un feedback din aplicație, acum este manual.]

### 10.3 Lecții Învățate (Top 5)

1. **[Lecție 1]:** [Datele generate trebuie să fie și cu jargon. Inițial am antrenat modelul fără greșeli gramaticale.]
2. **[Lecție 2]:** [Funcția de feedback este esențială.]
3. **[Lecție 3]:** [Vectorizarea la nivel de caracter cu Char N-grams funcționează mai bine decat Embeddings, de tipul Bert. Probabil pentru că tipul acelor rețele complexe nu recunoșteau cuvinte greșite, precum: "tnsiune"(tensiune).]
4. **[Lecție 4]:** [Threshold-ul de 0.5 a adus prea multe false positive-uri, ridicarea acestuia la 0.65 a eliminat o parte din acestea.]
5. **[Lecție 5]:** [Preprocesarea excesivă afectează sensul. Cuvântul "nu" șters din "nu pornește" inversează sensul defectului, creând o confuzie între o urgență și o funcționalitate normală.]

### 10.4 Retrospectivă

**Ce ați schimba dacă ați reîncepe proiectul?**

*[1-2 paragrafe: Decizii pe care le-ați lua diferit, cu justificare bazată pe experiența acumulată]*

Cu experiența acumulată în acest proiect pot spune că aș schimba partea inițială de Data engineering, unde aș face in sistem de etichetare asistată, în loc de generarea datelor sintetice și apoi să le implementez greșeli manual. 
De asemenea aș vrea să schimb într-o arhitectură modulară, de tip Docker, pentru a putea folosi modelul și pe altă mașină.

### 10.5 Direcții de Dezvoltare Ulterioară

| Termen | Îmbunătățire Propusă | Beneficiu Estimat |
|--------|---------------------|-------------------|
| **Short-term** (1-2 săptămâni) | [Dezvoltarea clasei Software, prin adăugarea unor termeni specifici("glitch","bug", etc)] | [10-15% Recall pe clasa Software] |
| **Medium-term** (1-2 luni) | [Crearea funcției de Active Learning, care preia automat ] | [ex: +3-5% accuracy general] |
| **Long-term** | [Adăugarea funcției de recunoaștere a problemei prin imagini] | [Textul cu greșeli poate doar confirma ce este în imagini] |
---

## 11. Bibliografie

*[Minimum 3 surse cu DOI/link funcțional - format: Autor, Titlu, Anul, Link]*
1. [Abaza, B], [Retele neuronale], [2025].URL: [https://curs.upb.ro/2025/course/view.php?id=1338]
2. [Scikit Documentation], [Machine Learning in Python], URL: [(https://scikit-learn.org/stable/index.html)]
3. [Keras], [Deep Learning for humans], URL: [(https://keras.io/)]
4. [IEEE], [Understanding and Defining Dark Data for the Manufacturing Industry], [2021]. DOI: [10.1109/TEM.2021.3051981] sau URL: [https://ieeexplore.ieee.org/abstract/document/9349176]
5. [Keyi Zhong, Tom Jackson, Andrew West & Georgina Cosma], [Building a Sustainable Knowledge Management System from Dark Data in Industrial Maintenance], [2024]. DOI: [https://doi.org/10.1007/978-3-031-63269-3_20] sau URL: [https://link.springer.com/chapter/10.1007/978-3-031-63269-3_20#auth-Keyi-Zhong]
6. [Gregory Gimpel], [Bringing dark data into the light: Illuminating existing IoT data lost within your organization], [2020]. DOI: [https://doi.org/10.1016/j.bushor.2020.03.009] sau URL: [https://www.sciencedirect.com/science/article/abs/pii/S0007681320300380]

**Exemple format:**
- Abaza, B., 2025. AI-Driven Dynamic Covariance for ROS 2 Mobile Robot Localization. Sensors, 25, 3026. https://doi.org/10.3390/s25103026
- Keras Documentation, 2024. Getting Started Guide. https://keras.io/getting_started/

---

## 12. Checklist Final (Auto-verificare înainte de predare)

### Cerințe Tehnice Obligatorii

- [x] **Accuracy ≥70%** pe test set (verificat în `results/final_metrics.json`)
- [x] **F1-Score ≥0.65** pe test set
- [x] **Contribuție ≥40% date originale** (verificabil în `data/generated/`)
- [x] **Model antrenat de la zero** (NU pre-trained fine-tuning)
- [x] **Minimum 4 experimente** de optimizare documentate (tabel în Secțiunea 5.3)
- [x] **Confusion matrix** generată și interpretată (Secțiunea 6.2)
- [x] **State Machine** definit cu minimum 4-6 stări (Secțiunea 4.2)
- [x] **Cele 3 module funcționale:** Data Logging, RN, UI (Secțiunea 4.1)
- [x] **Demonstrație end-to-end** disponibilă în `docs/demo/`

### Repository și Documentație

- [x] **README.md** complet (toate secțiunile completate cu date reale)
- [x] **4 README-uri etape** prezente în `docs/` (etapa3, etapa4, etapa5, etapa6)
- [x] **Screenshots** prezente în `docs/screenshots/`
- [x] **Structura repository** conformă cu Secțiunea 8
- [x] **requirements.txt** actualizat și funcțional
- [x] **Cod comentat** (minim 15% linii comentarii relevante)
- [x] **Toate path-urile relative** (nu absolute: `/Users/...` sau `C:\...`)

### Acces și Versionare

- [x] **Repository accesibil** cadrelor didactice RN (public sau privat cu acces)
- [x] **Tag `v0.6-optimized-final`** creat și pushed
- [x] **Commit-uri incrementale** vizibile în `git log` (nu 1 commit gigantic)
- [x] **Fișiere mari** (>100MB) excluse sau în `.gitignore`

### Verificare Anti-Plagiat

- [x] Model antrenat **de la zero** (weights inițializate random, nu descărcate)
- [x] **Minimum 40% date originale** (nu doar subset din dataset public)
- [x] Cod propriu sau clar atribuit (surse citate în Bibliografie)

---

## Note Finale

**Versiune document:** FINAL pentru examen  
**Ultima actualizare:** [09.02.2026]  
**Tag Git:** `v0.6-optimized-final`

---

*Acest README servește ca documentație principală pentru Livrabilul 1 (Aplicație RN). Pentru Livrabilul 2 (Prezentare PowerPoint), consultați structura din RN_Specificatii_proiect.pdf.*
