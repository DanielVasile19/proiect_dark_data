# 📘 README – Etapa 3: Analiza și Pregătirea Setului de Date pentru Rețele Neuronale

**Disciplina:** Rețele Neuronale  
**Instituție:** POLITEHNICA București – FIIR  
**Student:** Vasile Sorin-Daniel-Virgil  
**Data:** 21.11.2025

---

## Introducere

Acest document descrie activitățile realizate în **Etapa 3** pentru proiectul "SIA pentru Procesarea Dark Data în Mentenanța Industrială". Scopul este pregătirea unui set de date robust, capabil să antreneze o rețea neuronală să înțeleagă rapoarte de mentenanță scrise greșit (typos, argou industrial, lipsă diacritice), specifice unei fabrici de sticlă.

---

## 1. Structura Repository-ului Github (versiunea Etapei 3)
```
project-name/
├── README.md
├── requirements.txt
├── config/                         # Configurare erori (35%)
├── data/
│   ├── raw/rapoarte_mentenanta_v2.csv
│   ├── processed/                  # Artefacte (vectorizer, encoders)
│   ├── train/
│   ├── validation/
│   └── test/
├── docs/
│   └── datasets/                   # Grafice distribuție clase
└── src/
    ├── data_acquisition/generare_date_v2.py
    ├── neural_network/             # (Placeholder Etapa 4)
    └── preprocessing/              # Logica TF-IDF char-level
```
---

## 2. Descrierea Setului de Date

### 2.1 Sursa datelor

* **Origine:** Generare sintetică a datelor.
* **Modul de achiziție:** Prin script: generare_date_v2.py.
* **Justificare:** Datele de acest tip sunt confidențiale, am considerat că generarea sintetică este soluția. 

### 2.2 Caracteristicile dataset-ului

* **Număr total de observații:** 12,000 rapoarte.
* **Număr de caracteristici (features):** 1 (Text brut) -> transformat în vectori sparși (TF-IDF).
* **Tipuri de date:** ☑ Text (Input) / ☑ Categoriale (Output multiplu: Departament, Problemă, Urgență).
* **Format fișiere:** ☑ CSV (separator `|` pentru a permite virgule în text).

### 2.3 Descrierea fiecărei caracteristici

| **Caracteristică** | **Tip** | **Descriere** | **Exemplu Valori** |
|-------------------|---------|---------------|--------------------|
| `text_raport` | Text (Input) | Descrierea defectului scrisă de operator cu potențiale erori | "mtoor blocat la lina 2", "nu am tensine" |
| `eticheta_departament` | Categorial (Target 1) | Departamentul responsabil | {Mecanic, Electric, Software} |
| `eticheta_problema` | Categorial (Target 2) | Tipul specific de defect | {uzura_fizica, bug_aplicatie, senzori...} |
| `eticheta_urgenta` | Categorial (Target 3) | Nivelul de prioritate | {mică, medie, critica} |

---

## 3. Analiza Exploratorie a Datelor (EDA)

### 3.1 Statistici descriptive

* **Distribuție Departamente:** Balansat artificial (~33% fiecare) pentru a evita bias-ul inițial.
* **Lungime text:** Medie 45 caractere, Max 120 caractere (rapoarte scurte și concise).
* **Volum Dark Data:** Aproximativ 40% din cuvinte conțin cel puțin o eroare simulată (lipsă diacritice, typo-uri).

### 3.2 Analiza calității datelor

* **Valori lipsă:** 0% (Generatorul asigură completitudinea).
* **Consistență:** Datele sunt intenționat scrise greșit pentru ca modelul să se poată antrena luând în considerare și greșelile de scriere.
* **Corelații:** Există o corelație puternică între anumite cuvinte cheie (chiar și greșite, ex: "fmn", "fum") și eticheta `Electric` sau `Urgenta Critica`.

---

## 4. Preprocesarea Datelor

### 4.1 Curățarea și Transformarea

Având în vedere natura proiectului (procesare text scris greșit), preprocesarea clasică (stemming, lemmatization) **A FOST EXCLUSĂ** intenționat, deoarece ar distruge informația din cuvintele scrise greșit.

**Tehnica Aleasă: Character-level TF-IDF**
* **Vectorizare:** `TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4))`
* **De ce:** Această metodă sparge textul în n-grame de caractere.
    * Ex: "mtor" -> `['mt', 'to', 'or']`
    * Ex: "motor" -> `['mo', 'ot', 'to', 'or']`
* **Rezultat:** Rețeaua înțelege că "mtor" și "motor" sunt similare matematic, fără a avea nevoie de un dicționar corect.

### 4.2 Structurarea seturilor de date

**Împărțire:**
* 80% – Train (9,600 mostre)
* 20% – Test/Validation (2,400 mostre)

---

## 5. Fișiere Generate în Această Etapă

* `data/raw/rapoarte_mentenanta_v2.csv` – Datasetul complet cu zgomot.
* `models/vectorizer_v2.joblib` – Vectorizatorul antrenat (salvat pentru inferență).
* `models/encoder_*.joblib` – Encoderele pentru cele 3 etichete (Departament, Problema, Urgenta).

---

## 6. Stare Etapă

- [x] Structură repository configurată
- [x] Dataset generat cu algoritm de "human error simulation"
- [x] Preprocesare implementată (TF-IDF char-level)
- [x] Documentație actualizată