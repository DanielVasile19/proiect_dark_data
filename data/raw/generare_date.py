import pandas as pd
import random
import datetime

print("-------------------------------------------------")
print("🚀 [Pasul 1] Generare Date Sintetice (Dark Data)")
print("-------------------------------------------------")

# --- 1. Definirea "Bucăților" de propoziție ---

# Folosim "coduri" simple ca etichete
probleme_baza = {
    'motor_defect': "Motorul electric pare defect",
    'pompa_blocata': "Pompa hidraulică s-a blocat",
    'cordon_rupt': "Cordonul de alimentare este rupt",
    'scurgere_ulei': "Am observat o scurgere de ulei",
    'senzor_offline': "Un senzor este offline",
    'eroare_software': "Sistemul dă o eroare de software",
    'supraincalzire': "Componenta principală se supraîncălzește"
}

locatii_baza = {
    'axa_3': "în zona axei 3",
    'linia_1': "pe linia de producție 1",
    'robot_5': "la brațul robotic 5",
    'depozit': "în depozitul de piese",
    'panou_control': "la panoul de control principal"
}

urgente_baza = {
    'critica': "Necesită intervenție imediată. Linia este oprită.",
    'medie': "Trebuie verificat în următoarele 24 de ore. Producția e încetinită.",
    'mica': "De verificat la următoarea oprire planificată. Nu afectează producția."
}

# Transformăm dicționarele în liste pentru a le putea alege ușor
lista_probleme = list(probleme_baza.keys())
lista_locatii = list(locatii_baza.keys())
lista_urgente = list(urgente_baza.keys())

# --- 2. Logica de Generare ---

print(f"Se generează 1000 de rapoarte de mentenanță...")

data_generata = []  # Aici vom stoca toate rapoartele
numar_rapoarte = 1000

for i in range(numar_rapoarte):
    # Alegem aleatoriu câte o "bucată"
    eticheta_problema = random.choice(lista_probleme)
    eticheta_locatie = random.choice(lista_locatii)
    eticheta_urgenta = random.choice(lista_urgente)

    # Obținem textul corespunzător etichetelor
    text_problema = probleme_baza[eticheta_problema]
    text_locatie = locatii_baza[eticheta_locatie]
    text_urgenta = urgente_baza[eticheta_urgenta]

    # Asamblăm "Dark Data" (textul liber)
    text_raport_final = f"{text_problema} {text_locatie}. {text_urgenta}"

    # Adăugăm și o dată aleatorie pentru realism
    data_raport = datetime.date(2024, 1, 1) + datetime.timedelta(days=random.randint(0, 700))

    # Stocăm rezultatul (atât textul, cât și etichetele!)
    data_generata.append({
        'data_raport': data_raport,
        'text_raport': text_raport_final,
        'eticheta_problema': eticheta_problema,
        'eticheta_locatie': eticheta_locatie,
        'eticheta_urgenta': eticheta_urgenta
    })

print(f"✅ Au fost generate {len(data_generata)} rapoarte.")

# --- 3. Salvarea în fișier .csv ---

# Convertim lista noastră de dicționare într-un tabel Pandas
df = pd.DataFrame(data_generata)

# Salvăm tabelul într-un fișier CSV
NUME_FISIER_CSV = "rapoarte_mentenanta.csv"
df.to_csv(NUME_FISIER_CSV, index=False, encoding='utf-8-sig')

print(f"✅ Datele au fost salvate cu succes în '{NUME_FISIER_CSV}'")
print("\n--- Exemplu de 3 rânduri generate: ---")
print(df.head(3))