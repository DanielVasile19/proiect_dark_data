# 📘 README – Etapa 4: Arhitectura Completă a Aplicației SIA bazată pe Rețele Neuronale

**Disciplina:** Rețele Neuronale  
**Instituție:** POLITEHNICA București – FIIR  
**Student:** Vasile Sorin-Daniel-Virgil
**Data:** 11.12.2025

---

## Scopul Etapei 4

Dezvoltarea scheletului funcțional al aplicației de dispecerat automatizat pentru fabrica de sticlă, capabil să preia un text (chiar și scris greșit), să îl proceseze printr-o Rețea Neuronală Multi-Task și să afișeze rezultatele (Departament, Problemă, Urgență) într-o interfață grafică.

---

## 1. Tabelul Nevoie Reală → Soluție SIA → Modul Software

| **Nevoie reală concretă** | **Cum o rezolvă SIA-ul vostru** | **Modul software responsabil** |
|---------------------------|--------------------------------|--------------------------------|
| Reducerea timpului de triere a tichetelor de mentenanță (acum: 5-10 min/tichet) | Clasificare automată instantanee (<50ms) a tichetului către departamentul corect | **Modul RN (Multi-Task Model)** |
| Gestionarea rapoartelor scrise greșit de operatori ("Dark Data") | Utilizarea vectorizării la nivel de caracter (Char N-Grams) pentru a înțelege textul viciat | **Modul Preprocessing** |
| Evitarea deplasării echipelor greșite la intervenție (ex: Electrician la problemă Mecanică) | Predicția simultană a Departamentului și a Tipului de Problemă cu acuratețe ridicată | **Web Service / UI (Dashboard)** |

---

## 2. Contribuția Voastră Originală la Setul de Date

### Contribuția originală la setul de date:

**Total observații finale:** 12,000
**Observații originale:** 12,000 (100%)

**Tipul contribuției:**
[x] Date generate prin simulare (Scripting cu reguli de domeniu și injecție de erori)

**Descriere detaliată:**
Deoarece datele reale din fabrică nu pot fi scoase din rețeaua internă, am dezvoltat un generator de date (`src/data_acquisition/generare_date_v2.py`) care simulează comportamentul operatorilor. 
Scriptul folosește un dicționar ierarhic de defecțiuni reale (Mecanic/Electric/Software) și aplică o funcție de **"Chaos Monkey"** care introduce erori specifice tastării rapide:
1. Eliminare diacritice.
2. Inversare litere ("mtoor").
3. "Fat finger error" (apăsarea tastelor vecine pe QWERTY).
4. Omiterea literelor ("presine" vs "presiune").

**Locația codului:** `src/data_acquisition/generare_date_v2.py`
**Locația datelor:** `data/raw/rapoarte_mentenanta_v2.csv`

---

## 3. Diagrama State Machine a Întregului Sistem

**Justificarea State Machine-ului ales:**
Am ales o arhitectură de tip **"Event-Driven Classification"**. Sistemul stă în IDLE până când un operator introduce o sesizare. Starea critică este `HUMAN_LOOP_CORRECTION`: dacă AI-ul are o încredere scăzută sau greșește, operatorul poate corecta manual, iar datele sunt salvate pentru re-antrenare viitoare (Active Learning).

**Descriere Flux:**
`IDLE` → `INPUT_TEXT` → `PREPROCESS (Vectorizare Char-WB)` → `RN_INFERENCE (3 Heads)` → `DISPLAY_RESULT` → 
   ├─ [Corect] → `LOG_AUTO` → `IDLE`
   └─ [Incorect/Low Conf] → `HUMAN_CORRECTION` → `SAVE_TO_DATASET` → `IDLE`

*(Notă: Diagrama vizuală se găsește în `docs/state_machine.png`)*

---

## 4. Scheletul Complet al celor 3 Module

### Modul 1: Data Logging / Acquisition (`generare_date_v2.py`)
* **Funcționalitate:** Generează CSV-ul cu date de antrenare, aplicând logica de erori umane.
* **Status:** Funcțional. Rulează fără erori și produce fișierul cu separator Pipe (`|`) pentru a proteja virgulele din text.

### Modul 2: Neural Network Module (`train_model.py` & `model.py`)
* **Arhitectură:** Rețea Neuronală cu **Multi-Task Learning**.
    * Input: Vector TF-IDF (3000 features).
    * Shared Layers: Dense(128) -> Dropout -> Dense(64).
    * Outputs: 3 straturi Softmax separate (Problemă, Departament, Urgență).
* **Status:** Definit și compilat. Modelul este salvat în `models/trained_model.h5`.

### Modul 3: Web Service / UI (`dashboard.py`)
* **Tehnologie:** Streamlit.
* **Funcționalitate:**
    * Permite introducerea textului de avarie.
    * Afișează predicțiile în timp real (<50ms).
    * Include secțiune de **Feedback (Human-in-the-loop)** pentru corectarea predicțiilor greșite.
* **Status:** Funcțional. Interfața rulează în browser.

---

## Structura Repository-ului la Finalul Etapei 4
```
proiect-rn/
├── data/raw/rapoarte_mentenanta_v2.csv
├── src/
│   ├── app/dashboard.py
│   ├── data_acquisition/generare_date_v2.py
│   ├── neural_network/train_model.py
│   └── preprocessing/vectorizer_v2.joblib
├── models/trained_model.h5
├── docs/
│   ├── screenshots/ui_demo.png
│   └── state_machine.png
├── README_Etapa3.md
├── README_Etapa4_Arhitectura_SIA.md
└── requirements.txt
```
---

## Checklist Final

- [x] Tabelul Nevoie → Soluție completat.
- [x] Declarație contribuție 100% date originale (Generated Dark Data).
- [x] Cod generare date funcțional.
- [x] Diagrama State Machine descrisă.
- [x] Modul UI (Streamlit) funcțional și integrat cu modelul.