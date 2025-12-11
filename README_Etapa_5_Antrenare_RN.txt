# 📘 README – Etapa 5: Configurarea și Antrenarea Modelului RN

**Disciplina:** Rețele Neuronale  
**Instituție:** POLITEHNICA București – FIIR  
**Student:** Vasile Sorin-Daniel-Virgil  
**Link Repository GitHub:** https://github.com/DanielVasile19/proiect_dark_data  
**Data predării:** 11.12.2025

---

## Scopul Etapei 5

Această etapă corespunde punctului **6. Configurarea și antrenarea modelului RN** din lista de 9 etape.

**Obiectiv principal:** Antrenarea efectivă a modelului RN Multi-Task definit în Etapa 4, evaluarea performanței pe un set de test independent și integrarea modelului antrenat.

**Pornire obligatorie:** Arhitectura completă și funcțională din Etapa 4:
- State Machine definit și justificat
- Cele 3 module funcționale (Data Logging, RN, UI)
- Minimum 40% date originale în dataset (realizat: 100% date generate sintetic)

---

## PREREQUISITE – Verificare Etapa 4 (OBLIGATORIU)

**Înainte de a începe Etapa 5, s-au verificat livrabilele din Etapa 4:**

- [x] **State Machine** definit și documentat în `docs/state_machine.png`
- [x] **Contribuție ≥40% date originale** în `data/generated/` (Dataset complet de 5000 înregistrări)
- [x] **Modul 1 (Data Logging)** funcțional - produce CSV-uri (`generare_date_v2.py`)
- [x] **Modul 2 (RN)** cu arhitectură definită
- [x] **Modul 3 (UI/Web Service)** funcțional
- [x] **Tabelul "Nevoie → Soluție → Modul"** complet în README Etapa 4

---

## Pregătire Date pentru Antrenare

Deoarece s-a extins volumul de date în Etapa 4 la 5000 de înregistrări, procesul de preprocesare a fost rulat integral pe noul dataset.

**Parametri de preprocesare utilizați:**
- **Vectorizare:** TF-IDF (n-grams character level 3-5 chars, max features 7000).
- **Split:** 70% Train / 15% Validation / 15% Test.
- **Random State:** 42 pentru reproductibilitate.
- **Stratificare:** Aplicată pe clasa principală pentru a asigura distribuția echilibrată.

**Verificare dataset:**
- Total sample-uri: 5000
- Train: 3500
- Validation: 750
- Test: 750

---

## Cerințe Structurate pe 3 Niveluri

### Nivel 1 – Obligatoriu pentru Toți

1. **Antrenare model:** Modelul Multi-Task a fost antrenat pe setul final de 5000 date sintetice.
2. **Configurare:** Antrenarea a rulat pentru 30 epoci, cu mecanism de oprire automată.
3. **Împărțire stratificată:** Respectată (70/15/15).
4. **Metrici calculate pe test set:**
   - **Acuratețe:** 1.0000 (100%)
   - **F1-score (macro):** 1.0000 (100%)
5. **Salvare model:** Modelul final este salvat în `models/trained_model.h5`.
6. **Integrare UI:** Interfața încarcă acum modelul antrenat și realizează inferențe reale (demonstrat în screenshot).

#### Tabel Hiperparametri și Justificări

| **Hiperparametru** | **Valoare Aleasă** | **Justificare** |
|--------------------|-------------------|-----------------|
| **Learning rate** | 0.001 | Valoare standard pentru optimizatorul Adam, asigurând o convergență rapidă și stabilă pe date sparse (TF-IDF). |
| **Batch size** | 32 | Echilibru optim între utilizarea memoriei și stabilitatea gradientului pentru un dataset de 5000 samples. |
| **Number of epochs** | 30 | Număr suficient pentru a atinge convergența, controlat de Early Stopping pentru a preveni overfitting-ul. |
| **Optimizer** | Adam | Eficiență superioară pentru probleme NLP cu vectori rari, datorită ajustării automate a ratei de învățare per parametru. |
| **Loss function** | Sparse Categorical Crossentropy | Adecvată pentru clasificare multi-class cu etichete integer (nebinarizate) pe 3 ieșiri distincte. |
| **Activation functions** | ReLU (hidden), Softmax (output) | ReLU previne vanishing gradient în straturile dense; Softmax este necesar pentru distribuția probabilistică pe clasele de ieșire. |

---

### Nivel 2 – Recomandat

S-au implementat următoarele optimizări:

1. **Early Stopping:** Monitorizarea metricii `val_loss` cu `patience=5`. Antrenarea s-a oprit automat și a restaurat cei mai buni ponderi (`restore_best_weights=True`).
2. **Augmentări relevante:** Generarea datelor sintetice a inclus variații de topică (inversiuni subiect-predicat) și introducerea de greșelilor de scriere pentru a robustiza modelul la input uman imperfect.
3. **Grafic loss și val_loss:** Salvat în `docs/loss_curve.png`. Curbele indică o convergență rapidă și stabilă, fără divergențe majore între antrenare și validare.

**Indicatori obținuți:**
- **Acuratețe:** 100% (Target Nivel 2: ≥ 75%)
- **F1-score:** 1.00 (Target Nivel 2: ≥ 0.70)

---

## Verificare Consistență cu State Machine (Etapa 4)

Antrenarea și inferența respectă fluxul definit în State Machine:

| **Stare din Etapa 4** | **Implementare în Etapa 5** |
|-----------------------|-----------------------------|
| `ACQUIRE_DATA` | Generatorul `generare_date_v2.py` produce datele brute pentru antrenare. |
| `PREPROCESS` | Vectorizatorul TF-IDF antrenat (`vectorizer_v2.joblib`) este aplicat pe input-ul utilizatorului în UI. |
| `RN_INFERENCE` | Se apelează `model.predict()` folosind `trained_model.h5` (modelul real, nu dummy). |
| `DISPLAY` | Rezultatele (Problemă, Departament, Urgență) sunt afișate în Dashboard cu scoruri de încredere reale. |
| `FEEDBACK_LOOP` | Corecțiile manuale sunt salvate în dataset pentru cicluri viitoare de antrenare. |

Analiză Erori în Context Industrial (Nivel 2)
Deși pe setul de test sintetic performanța este maximă (datorită consistenței regulilor de generare), într-un mediu industrial real anticipăm următoarele provocări:

1. Pe ce clase greșește cel mai mult modelul?
Potențiale confuzii între "Eroare Software" și "Eroare HMI". Cauză: Suprapunere semantică mare (ambele implică ecrane/interfețe). Operatorii pot descrie o eroare de interfață (HMI) ca fiind o "eroare de soft".

2. Ce caracteristici ale datelor cauzează erori?
Textele extrem de scurte (ex: "defect", "nu merge") sau ambigue. Exemplu: "Linia s-a oprit". Poate fi o cauză mecanică, electrică sau de siguranță. Fără context suplimentar, TF-IDF nu poate extrage trăsături suplimentare.

3. Ce implicații are pentru aplicația industrială?
False Positives (Alarmă falsă): Acceptabil.

Misclassification (Departament greșit): Critic. Trimiterea unui electrician la o problemă mecanică (ex: scurgere ulei) duce la creșterea timpului de staționare (downtime).

Prioritate: Maximizarea preciziei pe clasa "Departament" pentru a asigura rutarea corectă a personalului.

4. Ce măsuri corective propuneți?
Human-in-the-Loop: Menținerea funcționalității de feedback din UI pentru a colecta date reale și a re-antrena modelul periodic.

Extinderea Vocabularului: Adăugarea continuă de termeni de argou specifici fabricii în setul de antrenare.

Prag de Siguranță: Implementarea unui confidence threshold. Dacă încrederea predicției este < 70%, sistemul să solicite operatorului să selecteze manual departamentul.
Structura arhitecturii proiectului:
proiect_dark_data/
├── README.md                        
├── README_Etapa4_Arhitectura_SIA.txt 
├── README_Etapa5_Antrenare_RN.md    
│
├── docs/
│   ├── state_machine.png            
│   ├── loss_curve.png               # Graficul curbelor de antrenare 
│   └── screenshots/
│       ├── ui_demo.png              # UI Demo 
│       └── inference_real.png       
│
├── data/
│   ├── raw/                        
│   ├── generated/                   
│   ├── processed/                   
│
├── src/
│   ├── data_acquisition/            
│   ├── neural_network/
│   │   ├── train_model.py           
│   │   └── predict_dispecer.py      
│   └── app/
│       └── dashboard.py             # UI Streamlit
│
├── models/
│   ├── trained_model.h5             # Modelul antrenat
│   └── *.joblib                     # Encoderele și vectorizatorul
│
├── results/
│   ├── training_history.csv         
│   └── test_metrics.json            # Rezultate finale
│
├── config/
└── requirements.txt