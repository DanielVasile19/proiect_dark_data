# 📘 README – Etapa 4: Arhitectura Completă a Aplicației SIA bazată pe Rețele Neuronale

**Disciplina:** Rețele Neuronale  
**Instituție:** POLITEHNICA București – FIIR  
**Student:** Vasile Sorin-Daniel-Virgil  
**Link Repository GitHub:** [Adaugă Link-ul Tău Aici]  
**Data:** 05.12.2025  

---

## Scopul Etapei 4

În această etapă se livrează scheletul complet și funcțional al sistemului **"Analiza si Clasificarea Automata a Rapoartelor de Mentenanta Industriala "**. Sistemul este capabil să genereze date, să definească o arhitectură de Rețea Neuronală Multi-Task și să ruleze un flux complet de la input-ul utilizatorului până la output și colectarea feedback-ului.

---

## 1. Tabelul Nevoie Reală → Soluție SIA → Modul Software

| **Nevoie reală concretă** | **Cum o rezolvă SIA-ul vostru** | **Modul software responsabil** |
|---------------------------|--------------------------------|--------------------------------|
| **Reducerea timpului de triaj** al rapoartelor de mentenanță scrise manual (text liber) | Clasificare automată a textului în < 1 secundă pentru identificarea problemei și departamentului. | **Modul 2 (RN)** + **Modul 3 (UI)** |
| **Eliminarea erorilor de alocare** (ex. trimiterea unui mecanic la o problemă software) | Predicție Multi-Task cu acuratețe țintă > 85% pentru rutarea tichetului către departamentul corect. | **Modul 2 (RN - Multi-Output)** |
| **Adaptarea la jargonul specific** și erorile de scriere ale operatorilor | Antrenare continuă pe date corectate de om, salvând feedback-ul pentru re-antrenare. | **Modul 3 (UI - Feedback)** |

---

## 2. Contribuția Originală la Setul de Date

### Declarație contribuție:

**Total observații finale:** 5000 (după Etapa 3 + Etapa 4)  
**Observații originale:** 5000 (100%)

**Tipul contribuției:**
[X] Date generate prin simulare (Generare Sintetică Programatică)  
[ ] Date achiziționate cu senzori proprii  
[ ] Etichetare/adnotare manuală  
[ ] Date sintetice prin metode avansate  

**Descriere detaliată:**
Proiectul utilizează o abordare de **Generare de Date Sintetice** pentru a simula scenarii industriale reale. Deoarece datele reale de mentenanță sunt confidențiale, am dezvoltat un generator propriu care:
1.  Combină un vocabular tehnic extins (defecte mecanice, electrice, software).
2.  Introduce greșeli specifice factorului uman: erori de scriere, variații de topică, abrevieri și jargon informal.
3.  Generează automat etichetele corecte (Problemă, Departament, Urgență) pe baza unor reguli de business predefinite.

Această metodă asigură un dataset perfect balansat și etichetat corect, esențial pentru antrenarea supravegheată a modelului Multi-Task.

**Locația codului:** `src/data_acquisition/generare_date_v2.py`  
**Locația datelor:** `data/generated/rapoarte_mentenanta_v2.csv`

**Dovezi:**
- Scriptul de generare este funcțional și parametrizabil.
- Dataset-ul rezultat conține variații complexe de text ("mtoor ars" vs "motor defect").
- Analiza exploratorie (EDA) disponibilă în `docs/datasets/`.

---

## 3. Diagrama State Machine a Întregului Sistem

**Locația diagramei:** `docs/state_machine.png`

### Justificarea State Machine-ului ales:

Am ales o arhitectură de tip feedback deoarece în domeniul mentenanței industriale, expertiza umană este critică, iar un model AI nu poate fi lăsat să ia decizii autonome de alocare a resurselor fără posibilitatea de corecție.

**Stările principale sunt:**
1.  **IDLE:** Sistemul așteaptă input de la operator.
2.  **PREPROCESS:** Textul brut introdus este curățat și vectorizat (TF-IDF).
3.  **INFERENCE:** Rețeaua Neuronală Multi-Task prezice simultan 3 valori (Problemă, Departament, Urgență).
4.  **DISPLAY_RESULTS:** Afișarea predicțiilor și a scorului de încredere.
5.  **WAIT_FEEDBACK:** Sistemul așteaptă validarea umană.

**Tranzițiile critice sunt:**
- De la **DISPLAY_RESULTS** la **SAVE_CORRECTION**: Aceasta este inovația sistemului. Dacă operatorul observă o eroare, corectează etichetele prin interfață, iar sistemul salvează noua pereche (Text, Etichete Corecte) în baza de date pentru re-antrenare viitoare.
- Starea **ERROR** este gestionată pentru a preveni blocarea aplicației în cazul unor input-uri invalide.

---

## 4. Scheletul Complet al celor 3 Module

### Modul 1: Data Logging / Acquisition
* **Locație:** `src/data_acquisition/generare_date_v2.py`
* **Descriere:** Script Python care generează 5000+ rapoarte sintetice. Include dicționare de termeni tehnici și logică de randomizare pentru a simula erori umane.
* **Status:** Funcțional. Generează fișierul CSV în `data/generated/`.

### Modul 2: Neural Network Module
* **Locație:** `src/neural_network/train_model.py`
* **Descriere:** Definește și compilează o Rețea Neuronală de tip **Multi-Layer Perceptron (MLP)** cu arhitectură **Multi-Output** (3 capete de ieșire softmax independente).
* **Arhitectură:**
    * Input Layer: Vector TF-IDF.
    * Hidden Layers: Dense (128) -> Dropout -> Dense (64).
    * Output Layers: 3 straturi Dense separate pentru cele 3 target-uri (Problemă, Departament, Urgență).
* **Status:** Definit, compilat, cu funcționalitate de salvare/încărcare în folderul `models/`.

### Modul 3: Web Service / UI
* **Locație:** `src/app/dashboard.py`
* **Tehnologie:** Streamlit.
* **Descriere:** Interfață web care permite operatorului să introducă text, vizualizează predicțiile modelului și oferă o formă de feedback pentru corecție manuală. Include și un dashboard de analiză statistică a datelor istorice.
* **Status:** Funcțional end-to-end. Primește input, rulează inferența și salvează feedback-ul.

---

## Structura Repository-ului (Final Etapa 4)

```text
proiect_dark_data/
├── data/
│   ├── raw/               # Date brute (istorice)
│   ├── processed/         # Date procesate
│   ├── generated/         # Datele originale (generare_date_v2.py output)
│   ├── train/             # (Seturi interne)
│   ├── validation/        # (Seturi interne)
│   └── test/              # (Seturi interne)
├── src/
│   ├── data_acquisition/  # Modul 1: generare_date_v2.py
│   ├── preprocessing/     # Funcții TF-IDF (incluse în pipeline)
│   ├── neural_network/    # Modul 2: train_model.py, predict_dispecer.py
│   └── app/               # Modul 3: dashboard.py
├── docs/
│   ├── datasets/          # Grafice EDA
│   ├── state_machine.png  # Diagrama stărilor (OBLIGATORIU)
│   └── screenshots/       # Capturi ecran UI
├── models/                # Modelul .keras și encoderele .joblib
├── config/                # Fișiere configurare
├── README.md              # Readme general
├── README_Etapa3.md       # Readme anterior
├── README_Etapa4_Arhitectura_SIA.md # Acest fișier (Livrabil Etapa 4)
└── requirements.txt       # Dependențe