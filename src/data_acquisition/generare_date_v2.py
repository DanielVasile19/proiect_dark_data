import os
import random
import datetime
import pandas as pd
import unicodedata

# --- CONFIGURARE ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'raw')
os.makedirs(DATA_DIR, exist_ok=True)
OUTPUT_FILE = os.path.join(DATA_DIR, 'rapoarte_mentenanta_v2.csv')

# Generam destule date pentru ca reteaua sa invete variatiile de greseli
DATASET_SIZE = 12000

# --- IERARHIE DEPARTAMENT -> PROBLEME -> SIMPTOME ---
DATA_MAPPING = {
    'Mecanic': {
        'uzura_fizica': [
            "curea rupta", "dinti rupti la roata dintata", "lant rupt", "banda transportoare sfiasiata",
            "rulment spart", "zgomot de polizare", "vibratii puternice la ax", "se aude urat un motor",
            "bataie in reductor", "ax descentrat"
        ],
        'blocaj_mecanic': [
            "axul motorului e blocat", "valva hidraulica blocata", "pistonul nu mai revine",
            "role blocate la banda", "usa de protectie intepenita", "angrenaj gripat",
            "s-a blocat bratul robotic", "nu mai culiseaza sania"
        ],
        'scurgere_lichide': [
            "scurgere de ulei sub utilaj", "pierde lichid hidraulic", "pata mare de ulei pe jos",
            "furtun presiune spart", "picura apa din racire", "vaselina curge din lagar"
        ]
    },
    'Electric': {
        'alimentare_curent': [
            "nu avem tensiune la panou", "siguranta sarita", "sursa de alimentare arsa",
            "scurtcircuit cu flama", "s-a oprit curentul la linie", "cablu sectionat",
            "nu se aprinde becul de control", "fluctuatii de tensiune"
        ],
        'senzori_instrumentatie': [
            "senzorul nu citeste nimic", "fotocelula murdara", "encoderul da eroare pozitie",
            "valoare zero la senzor temperatura", "termocupla defecta", "senzor proximitate lovit",
            "eroare citire debitmetru"
        ],
        'componente_automatizare': [
            "motorul declanseaza protectia termica", "releu ars in tablou", "invertorul e in avarie",
            "contactul nu cupleaza", "transformator supraincalzit", "variatorul de turatie e rosu"
        ]
    },
    'Software': {
        'bug_aplicatie': [
            "eroare pe ecran", "cod de eroare 404", "bug in soft la salvare",
            "nu se incarca reteta", "blue screen la pornire", "aplicatia s-a inchis singura",
            "datele nu se actualizeaza", "butonul de start e gri (inactiv)"
        ],
        'conectivitate_retea': [
            "nu comunica cu serverul", "pierdere pachete date", "nu avem retea la hmi",
            "cablu de retea scos", "adresa IP conflict", "ping timeout", "nu merge internetul"
        ],
        'hardware_it': [
            "touchscreen-ul nu raspunde la atingere", "ecran negru complet",
            "touchscreen decalat", "mouse-ul nu merge", "tastatura blocata",
            "unitatea PC caraie", "nu vede stick-ul USB"
        ]
    }
}

LOCATII = [
    "linia 1", "linia 2", "zona ambalare", "robot kuka", "robot sudura",
    "statia vopsire", "magazie piese", "presa hidraulica", "cnc mazak",
    "cuptor tratament", "camera compresoare", "zona expeditie"
]

URGENTE = {
    'critica': ["URGENT", "OPRITI TOT", "PERICOL", "FUM", "LINIE OPRITA", "INCENDIU", "RISC ACCIDENT"],
    'medie': ["trebuie verificat", "merge greu", "incetineste productia", "zgomot ciudat", "verificati cand puteti"],
    'mica': ["cand se poate", "nu e graba", "mentenanta preventiva", "verificare vizuala", "nota informativa"]
}

# --- LOGICA "DARK DATA" (SIMULARE GRESELI) ---
TASTATURA_VECINI = {
    'a': 'qwsz', 'b': 'vghn', 'c': 'xdfv', 'd': 'serfc', 'e': 'wsdr',
    'f': 'drtgv', 'g': 'ftyhb', 'h': 'gyujn', 'i': 'ujko', 'j': 'huikm',
    'k': 'jiolm', 'l': 'kop', 'm': 'njk', 'n': 'bhjm', 'o': 'iklp',
    'p': 'ol', 'q': 'aw', 'r': 'edft', 's': 'awedx', 't': 'rfgy',
    'u': 'yhji', 'v': 'cfgb', 'w': 'qase', 'x': 'zsdc', 'y': 'tghu', 'z': 'asx'
}


def sterge_diacritice(text):
    # In fabrica nimeni nu scrie cu "ș" sau "ț"
    text = unicodedata.normalize('NFD', text).encode('ascii', 'ignore').decode("utf-8")
    return text


def introduce_erori_umane(text, rata_eroare=0.35):
    """
    Simuleaza un muncitor grabit:
    - rata_eroare: probabilitatea ca un cuvant sa fie 'stricat'
    """
    # 1. Mai intai eliminam diacriticele (standard in industrie)
    text = sterge_diacritice(text)

    cuvinte = text.split()
    cuvinte_noi = []

    for cuv in cuvinte:
        # Nu stricam cuvintele foarte scurte (de, la, si) decat rar
        if len(cuv) < 3:
            cuvinte_noi.append(cuv)
            continue

        if random.random() < rata_eroare:
            # Alegem un tip de greseala
            tip = random.choice(['skip', 'swap', 'fat_finger', 'double'])

            chars = list(cuv)
            idx = random.randint(0, len(chars) - 1)

            try:
                if tip == 'skip' and len(chars) > 3:
                    # Mananca o litera: "motor" -> "mtor"
                    chars.pop(idx)

                elif tip == 'swap' and len(chars) > 2 and idx < len(chars) - 1:
                    # Inverseaza litere: "motor" -> "motoor" sau "mtoor"
                    chars[idx], chars[idx + 1] = chars[idx + 1], chars[idx]

                elif tip == 'fat_finger':
                    # Apasa tasta gresita: "s" -> "a" sau "d"
                    char_curent = chars[idx].lower()
                    if char_curent in TASTATURA_VECINI:
                        chars[idx] = random.choice(TASTATURA_VECINI[char_curent])

                elif tip == 'double':
                    # Apasa tasta de doua ori: "bataie" -> "bataiee"
                    chars.insert(idx, chars[idx])

            except:
                pass  # Daca da eroare de index, lasam cuvantul asa

            cuvinte_noi.append("".join(chars))
        else:
            cuvinte_noi.append(cuv)

    # Uneori operatorii uita spatiile dupa virgula
    rezultat = " ".join(cuvinte_noi)
    if random.random() < 0.2:
        rezultat = rezultat.replace(", ", ",")

    return rezultat.lower()  # Totul lowercase pentru consistenta


def formateaza_text(simptom, locatie, urgenta_text):
    sabloane = [
        f"{simptom} la {locatie}. {urgenta_text}",
        f"{urgenta_text}! {simptom} la {locatie}",
        f"La {locatie}: {simptom}.",
        f"Defectiune {locatie}: {simptom}.",
        f"Verificati {locatie}, {simptom}",
        f"{simptom} zona {locatie}"
    ]
    return random.choice(sabloane)


def genereaza_dataset(n):
    print(f"🔄 Generez {n} intrari cu simulare de erori umane (Dark Data)...")
    data = []
    departamente = list(DATA_MAPPING.keys())

    for _ in range(n):
        # 1. Selectie Logica
        dep = random.choice(departamente)
        eticheta_prob = random.choice(list(DATA_MAPPING[dep].keys()))
        simptom_baza = random.choice(DATA_MAPPING[dep][eticheta_prob])

        locatie = random.choice(LOCATII)
        eticheta_urg = random.choice(list(URGENTE.keys()))
        text_urg = random.choice(URGENTE[eticheta_urg])

        # 2. Formare Propozitie Corecta
        text_curat = formateaza_text(simptom_baza, locatie, text_urg)

        # 3. APLICARE "MURDARIE" (Dark Data)
        # Aplicam erori aleatorii pentru a antrena reteaua sa fie robusta
        text_final = introduce_erori_umane(text_curat, rata_eroare=0.4)

        # 4. Data
        days_offset = random.randint(0, 60)
        data_raport = (datetime.date.today() - datetime.timedelta(days=days_offset))

        data.append({
            'text_raport': text_final,  # Textul "murdar" pe care invata AI-ul
            'text_clean_debug': text_curat,  # Pastram si originalul pentru debugging daca e nevoie
            'eticheta_problema': eticheta_prob,
            'eticheta_departament': dep,
            'eticheta_urgenta': eticheta_urg,
            'data_raport': data_raport
        })

    return pd.DataFrame(data)


if __name__ == "__main__":
    df = genereaza_dataset(DATASET_SIZE)

    # SALVARE
    # Folosim Pipe (|) separator pentru ca textele pot contine virgule
    df.to_csv(OUTPUT_FILE, index=False, sep='|')

    print(f"✅ GATA! Fisier salvat: {OUTPUT_FILE}")
    print("\nExemple de date generate (observa greselile de scriere):")
    print("-" * 50)
    for i in range(5):
        row = df.iloc[i]
        print(f"Original: {row['text_clean_debug']}")
        print(f"DarkData: {row['text_raport']}")
        print("-" * 50)