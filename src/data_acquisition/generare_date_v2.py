import pandas as pd
import random
import datetime

print("-------------------------------------------------")
print("🚀 [Pasul 1.v2] Generare Date Sintetice")
print("-------------------------------------------------")


# Acum folosim liste de variații pentru a simula erorile umane

probleme_baza = {
    'motor_defect': ["Motorul electric pare defect", "probleme la motor", "motorul s-a oprit brusc", "mtoor ars",
                     "motorul face zgomot ciudat", "pb la motor", "nush ce nu merge aici la motor"],
    'pompa_blocata': ["Pompa hidraulică s-a blocat", "pompa nu mai merge", "presiune zero la pompa", "pompa e blocata", "nu mai pompeaza pompa"],
    'cordon_rupt': ["Cordonul de alimentare este rupt", "cablu de curent taiat", "cordonul e smuls",
                    "cablaj intrerupt"],
    'scurgere_ulei': ["Am observat o scurgere de ulei", "curge ulei pe jos", "balta de ulei sub utilaj",
                      "scurgere hidraulica", "balta ulei"],
    'senzor_offline': ["Un senzor este offline", "senzorul nu mai raporteaza", "eroare citire senzor", "snzor defect",
                       "senzorul e mort"],
    'eroare_software': ["Sistemul dă o eroare de software", "eroare soft", "programul a crapat", "sistem blocat",
                        "necesita restart software", "nu mai merge sistemul"],
    'supraincalzire': ["Componenta principală se supraîncălzește", "se incinge prea tare", "ATENTIE supraincalzire",
                       "temperatura e prea mare"]
}

locatii_baza = {
    'axa_3': ["în zona axei 3", "la axa 3", "pe bratul 3"],
    'linia_1': ["pe linia de producție 1", "la linia 1", "pe banda 1"],
    'robot_5': ["la brațul robotic 5", "pe robotul 5", "robot 5"],
    'depozit': ["în depozitul de piese", "langa depozit"],
    'panou_control': ["la panoul de control principal", "pe panoul de control", "la PLC"]
}

urgente_baza = {
    'critica': ["Necesită intervenție imediată. Linia este oprită.", "OPRIRE TOTALA. CRITIC.",
                "Urgent, linia blocata.", "rog rezolvarea imediata"],
    'medie': ["Trebuie verificat în următoarele 24 de ore. Producția e încetinită.", "De verificat azi.",
              "Incetineste productia."],
    'mica': ["De verificat la următoarea oprire planificată.", "Nu e urgent.",
             "De notat pentru mentenanta saptamanala."]
}

# --- 2. REGULA DE BUSINESS (Dispeceratul Inteligent) ---
# Aici definim legătura dintre problemă și departament
mapare_departament = {
    'motor_defect': 'Mecanic',
    'pompa_blocata': 'Mecanic',
    'cordon_rupt': 'Automatist',  # Problemă electrică
    'scurgere_ulei': 'Mecanic',
    'senzor_offline': 'Automatist',  # Problemă de senzorică/electr(on)ică
    'eroare_software': 'Software',  # Problemă de cod
    'supraincalzire': 'Mecanic'  # Problemă de răcire/frecare
}

# Transformăm cheile în liste pentru a le putea alege ușor
lista_probleme = list(probleme_baza.keys())
lista_locatii = list(locatii_baza.keys())
lista_urgente = list(urgente_baza.keys())

# --- 3. Logica de Generare (Modernizată) ---
print(f"Se generează 2000 de rapoarte de mentenanță (V2)...")
data_generata = []
numar_rapoarte = 2000  # Am mărit numărul

for i in range(numar_rapoarte):
    # Alegem aleatoriu ETICHETELE
    eticheta_problema = random.choice(lista_probleme)
    eticheta_locatie = random.choice(lista_locatii)
    eticheta_urgenta = random.choice(lista_urgente)

    # Aplicăm REGULA DE BUSINESS pentru a găsi departamentul
    eticheta_departament = mapare_departament[eticheta_problema]

    # Acum, alegem aleatoriu VARIAȚIILE DE TEXT
    text_problema = random.choice(probleme_baza[eticheta_problema])
    text_locatie = random.choice(locatii_baza[eticheta_locatie])
    text_urgenta = random.choice(urgente_baza[eticheta_urgenta])

    # Asamblăm "Dark Data" (textul "murdar" și variabil)
    text_raport_final = f"{text_problema} {text_locatie}. {text_urgenta}"

    # Adăugăm o dată aleatorie
    data_raport = datetime.date(2024, 1, 1) + datetime.timedelta(days=random.randint(0, 700))

    # Stocăm rezultatul (cu noua etichetă!)
    data_generata.append({
        'data_raport': data_raport,
        'text_raport': text_raport_final,
        'eticheta_problema': eticheta_problema,
        'eticheta_locatie': eticheta_locatie,
        'eticheta_urgenta': eticheta_urgenta,
        'eticheta_departament': eticheta_departament  # <-- NOUA COLOANĂ ȚINTĂ
    })

print(f"✅ Au fost generate {len(data_generata)} rapoarte.")

# --- 4. Salvarea în fișier .csv ---
df = pd.DataFrame(data_generata)
NUME_FISIER_CSV = "rapoarte_mentenanta_v2.csv"
df.to_csv(NUME_FISIER_CSV, index=False, encoding='utf-8-sig')

print(f"✅ Datele au fost salvate cu succes în '{NUME_FISIER_CSV}'")
print("\n--- Exemplu de 3 rânduri generate (V2): ---")
print(df.head(3))