# proiect_dark_data

**Disciplina:** Rețele Neuronale  
**Instituție:** POLITEHNICA București – FIIR  
**Student:** Vasile Sorin-Daniel-Virgil  
**Data:** 21/11/2025 

## Introducere

Acest document descrie activitățile realizate în Etapa 3, în cadrul proiectului Analiza 	si Clasificarea Automata a Rapoartelor de Mentenanta Industriala. Scopul etapei este generarea, analiza și preprocesarea unui set de date de rapoarte de mentenanță (text nestructurat) pentru antrenarea unui model de Rețea Neuronală Multi-Task.

##  1. Structura Repository-ului Github (versiunea Etapei 3)

proiect_dark_data/
├── README.md              # Documentația proiectului
├── requirements.txt       # Lista dependențelor (tensorflow, pandas, streamlit)
├── config/                # Fișiere de configurare (viitoare)
├── docs/
│   └── datasets/          # Documentație suplimentară despre date
├── data/
│   ├── raw/               # Date brute (rapoarte_mentenanta_v2.csv)
│   ├── processed/         # Modele salvate (.keras) și encodere (.joblib)
│   ├── train/             # (Seturi generate intern la rulare)
│   ├── validation/        # (Seturi generate intern la rulare)
│   └── test/              # (Seturi generate intern la rulare)
└── src/
    ├── preprocessing/     # Funcții de procesare text (TF-IDF)
    ├── data_acquisition/  # Scripturi generare date (generare_date_v2.py)
    └── neural_network/    # Scripturi antrenare și inferență (antrenare_date_Multitask.py)

##  2. Descrierea Setului de Date

### 2.1 Sursa datelor

* **Origine:** Date Sintetice generate, simulând rapoarte reale de mentenanță industrială ("Dark Data")
* **Modul de achiziție:** ☑ Generare programatică (Script Python: src/data_acquisition/generare_date_v2.py)
* **Perioada / condițiile colectării:** Datele simulează un istoric de mentenanță, incluzând intenționat erori umane (greșeli de scriere, jargon tehnic, exprimări colocviale) pentru a testa robustețea modelului.

### 2.2 Caracteristicile dataset-ului

* **Număr total de observații:** 3000
* **Număr de caracteristici (features):** 1 Input (Text) -> 3 Ieșiri (Etichete).
* **Tipuri de date:** ☑ Text (Natural Language) / ☑ Categoriale (Etichete)
* **Format fișiere:** ☑ CSV (rapoarte_mentenanta_v2.csv).

### 2.3 Descrierea fiecărei caracteristici

| **Caracteristică** | **Tip**  **Descriere** | **Domeniu valori** |
|-------------------|---------|-------------|---------------|--------------------|
|   text_raport     | text(string) | Descrierea defecțiunii introdusă de operator (input nestructurat). | 0–150 |
| eticheta_problema | categorial | Tipul tehnic al defecțiunii (Target 1). | {A, B, C} |
| eticheta_departament | categorial |Departamentul responsabil (Target 2)| 0–2.5 |
| eticheta_urgenta | categorial |Nivelul de prioritate (Target 3) | ... |

##  3. Analiza Exploratorie a Datelor (EDA) – Sintetic

### 3.1 Statistici descriptive aplicate

3.1 Statistici descriptive aplicate

Setul de date este balansat artificial prin generare randomizată uniformă a problemelor tehnice.

S-a asigurat o acoperire completă a tuturor scenariilor de defecțiune definite în regulile de business.

### 3.2 Analiza calității datelor

Valori lipsă: 0% (datele sunt generate controlat).

Consistență: Datele conțin intenționat "zgomot" (noise) sub formă de greșeli de scriere (ex: "snzor" vs "senzor") și variații de topică (ex: "motor defect la axa 3" vs "la axa 3 motorul nu merge").

### 3.3 Probleme identificate

#############################

##  4. Preprocesarea Datelor

### 4.1 Curățarea datelor

4.1 Curățarea datelor

Deoarece datele simulează text introdus de om, "curățarea" clasică este minimă; modelul este proiectat să învețe direct din textul "murdar".

Nu există valori lipsă (NaN) de tratat.

### 4.2 Transformarea caracteristicilor

Vectorizare Text (TF-IDF): Textul brut este transformat în vectori numerici folosind TfidfVectorizer cu n-grame la nivel de caracter (analyzer='char_wb', ngram_range=(3,5)). Aceasta permite recunoașterea cuvintelor chiar și când sunt scrise greșit.

Label Encoding: Etichetele categoriale (Problema, Departament, Urgență) sunt transformate în numere întregi (0, 1, 2...) folosind LabelEncoder din Scikit-Learn.

### 4.3 Structurarea seturilor de date

Împărțire recomandată (realizată automat în script):

80% – train (Set de instruire)

20% – test (Set de testare pentru evaluarea finală)

Principii respectate:

Stratificare pe eticheta principală pentru a asigura distribuția egală a claselor.

Vectorizatorul este antrenat doar pe setul de train pentru a evita data leakage.

### 4.4 Salvarea rezultatelor preprocesării

Obiectele de preprocesare (vectorizer, label_encoders) sunt salvate în format .joblib în folderul data/processed/ pentru a asigura consistența datelor în faza de inferență (în aplicația Dashboard).

Modelul antrenat este salvat ca .keras în același folder.

##  5. Fișiere Generate în Această Etapă

data/raw/rapoarte_mentenanta_v2.csv – date brute (generate).

data/processed/*.joblib – Encoderele și vectorizatorul antrenat.

data/processed/model_dispecer_v2.keras – Modelul RN antrenat.

src/neural_network/antrenare_date_Multitask.py – codul de preprocesare și antrenare.

##  6. Stare Etapă (de completat de student)

- [ ] Structură repository configurată
- [x] Dataset analizat (EDA realizată)
- [x] Date preprocesate
- [x] Seturi train/val/test generate
- [ ] Documentație actualizată în README + `data/README.md`
