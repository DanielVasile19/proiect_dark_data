import os
import random
import datetime
import pandas as pd

# Configurare cai
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'raw')
OUTPUT_FILE = os.path.join(DATA_DIR, 'rapoarte_mentenanta_v2.csv')

# Dictionare date sintetice
PROBLEME = {
    'motor_defect': ["motor ars", "zgomot motor", "vibratii puternice motor", "motorul s-a oprit", "supraincalzire motor"],
    'pompa_blocata': ["pompa nu trage", "presiune scazuta pompa", "pompa blocata", "scurgere la pompa"],
    'senzor_offline': ["senzorul nu raspunde", "eroare citire senzor", "senzor offline", "valoare zero senzor"],
    'eroare_software': ["eroare la plc", "ecran albastru", "bug software", "nu se conecteaza la retea", "eroare HMI"],
    'scurgere_ulei': ["ulei pe jos", "scurgere lichid hidraulic", "pata de ulei", "pierdere presiune ulei"]
}

LOCATII = ['linia 1', 'linia 2', 'zona ambalare', 'robot sudura', 'magazie piese', 'statia vopsire']

MAPARE_DEPARTAMENT = {
    'motor_defect': 'Mecanic',
    'pompa_blocata': 'Mecanic',
    'scurgere_ulei': 'Mecanic',
    'senzor_offline': 'Electric',
    'eroare_software': 'Software'
}

URGENTE = ['mica', 'medie', 'critica']

def genereaza_dataset(n=5000):
    data = []
    print(f"Generare {n} inregistrari...")

    for _ in range(n):
        # Selectie randomizata
        cheie_prob = random.choice(list(PROBLEME.keys()))
        text_prob = random.choice(PROBLEME[cheie_prob])
        locatie = random.choice(LOCATII)
        urgenta = random.choice(URGENTE)
        
        # Constructie text natural
        text_raport = f"{text_prob} la {locatie}. Urgenta este {urgenta}."
        
        # Adaugare in lista
        data.append({
            'text_raport': text_raport,
            'eticheta_problema': cheie_prob,
            'eticheta_departament': MAPARE_DEPARTAMENT[cheie_prob],
            'eticheta_urgenta': urgenta,
            'data_raport': datetime.date.today()
        })

    return pd.DataFrame(data)

if __name__ == "__main__":
    # Creare director daca nu exista
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # Executie generare
    df = genereaza_dataset(10000)
    
    # Export CSV
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Dataset salvat cu succes: {OUTPUT_FILE}")