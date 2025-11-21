import streamlit as st
import tensorflow as tf
import joblib
import numpy as np
import pandas as pd
import sys
import time
import datetime  # <-- PAS NOU: Adaugă acest import


# --- 1. ÎNCĂRCAREA RESURSELOR (Modele ȘI Date) ---

@st.cache_resource
def load_all_models():
    # ... (Funcția ta de încărcare a modelelor rămâne neschimbată) ...
    print("--- [Streamlit] Se încarcă modelele de pe disc... ---")
    try:
        model = tf.keras.models.load_model('model_dispecer_v2.keras')
        vectorizer = joblib.load('vectorizer_v2.joblib')
        encoder_problema = joblib.load('encoder_problema_v2.joblib')
        encoder_departament = joblib.load('encoder_departament_v2.joblib')
        encoder_urgenta = joblib.load('encoder_urgenta_v2.joblib')
        print("--- [Streamlit] Modele încărcate cu succes! ---")
        return model, vectorizer, encoder_problema, encoder_departament, encoder_urgenta
    except Exception as e:
        print(f"EROARE LA ÎNCĂRCAREA MODELELOR: {e}")
        st.error(f"EROARE CRITICĂ: Nu am putut încărca fișierele modelului. Detalii: {e}")
        return None, None, None, None, None


@st.cache_data
def load_csv_data(file_path):
    # ... (Funcția ta de încărcare CSV rămâne neschimbată) ...
    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        st.error(f"EROARE: Fișierul de date '{file_path}' nu a fost găsit.")
        return None


# Încărcăm resursele
(model, vectorizer, encoder_problema,
 encoder_departament, encoder_urgenta) = load_all_models()

df_analiza = load_csv_data("rapoarte_mentenanta_v2.csv")

# --- 2. CONSTRUIREA INTERFEȚEI (Cu Navigare) ---

st.sidebar.title("Navigație")
page = st.sidebar.radio(
    "Alegeți pagina:",
    ["Dispecerat Inteligent", "Dashboard Analiză"]
)
st.sidebar.markdown("---")
st.sidebar.info("Proiect realizat de Vasile Daniel - FIIR 2025")

# --- 3. LOGICA PAGINILOR ---

# --- PAGINA 1: DISPECERAT ---
if page == "Dispecerat Inteligent":

    st.title("🤖 Dispecerat Inteligent (Dark Data AI)")
    st.subheader("Analiză și rutare automată a rapoartelor de mentenanță")

    st.markdown("""
    Introduceți un raport de mentenanță în text liber. AI-ul va citi textul, 
    îl va clasifica și îl va aloca departamentului corect.
    """)

    text_nou_utilizator = st.text_area(
        "Introduceți raportul de mentenanță aici:",
        "mtoor ars pe linia 1. Urgent, linia blocata.",
        height=100
    )

    if st.button("Analizează Raportul"):
        if model is None or df_analiza is None:
            st.error("Modelul sau datele nu sunt încărcate. Verifică consola PyCharm.")
        elif not text_nou_utilizator.strip():
            st.warning("Te rog introdu un text în căsuță.")
        else:
            with st.spinner("AI-ul analizează textul..."):
                # ... (Logica de predicție rămâne identică) ...
                text_vec = vectorizer.transform([text_nou_utilizator])
                text_gata = text_vec.toarray()
                predictie_bruta = model.predict(text_gata, verbose=0)

                prob_problema = predictie_bruta[0][0]
                idx_problema = np.argmax(prob_problema)
                eticheta_problema = encoder_problema.inverse_transform([idx_problema])[0]
                scor_problema = prob_problema[idx_problema]

                prob_departament = predictie_bruta[1][0]
                idx_departament = np.argmax(prob_departament)
                eticheta_departament = encoder_departament.inverse_transform([idx_departament])[0]
                scor_departament = prob_departament[idx_departament]

                prob_urgenta = predictie_bruta[2][0]
                idx_urgenta = np.argmax(prob_urgenta)
                eticheta_urgenta = encoder_urgenta.inverse_transform([idx_urgenta])[0]
                scor_urgenta = prob_urgenta[idx_urgenta]

            st.success("Analiză completă!")
            st.subheader("Rezultatul Dispeceratului AI:")

            col1, col2, col3 = st.columns(3)
            # ... (Codul pentru st.metric rămâne neschimbat) ...
            col1.metric(label="🏷️ Problemă Identificată", value=eticheta_problema.replace("_", " ").title(),
                        help=f"Încredere: {scor_problema * 100:.2f}%")
            col2.metric(label="👨‍🔧 Departament Alocat", value=eticheta_departament.title(),
                        help=f"Încredere: {scor_departament * 100:.2f}%")
            col3.metric(label="⚠️ Urgență Stabilită", value=eticheta_urgenta.title(),
                        help=f"Încredere: {scor_urgenta * 100:.2f}%")

            # --- [BLOC NOU] Funcționalitatea "Human-in-the-Loop" ---
            st.markdown("---")
            st.warning("⚠️ Predicția AI a fost greșită?")

            with st.expander("Click aici pentru a corecta manual (Feedback)"):

                # Avem nevoie de listele de opțiuni pentru a popula dropdown-urile
                # Le luăm din encodere și din DataFrame-ul istoric
                options_problema = encoder_problema.classes_
                options_locatie = df_analiza['eticheta_locatie'].unique()  # Luăm locațiile din CSV
                options_departament = encoder_departament.classes_
                options_urgenta = encoder_urgenta.classes_

                with st.form(key="feedback_form"):
                    st.markdown("Vă rugăm selectați etichetele corecte pentru raportul de mai sus:")

                    # Câmpurile pentru corecție
                    col1_fb, col2_fb = st.columns(2)

                    eticheta_corecta_problema = col1_fb.selectbox(
                        "Problemă Corectă:", options=options_problema,
                        index=list(options_problema).index(eticheta_problema)  # Pre-selectează predicția AI
                    )

                    eticheta_corecta_locatie = col2_fb.selectbox(
                        "Locație Corectă:", options=options_locatie
                        # Nu putem pre-selecta locația, deoarece modelul nu o prezice (încă)
                    )

                    eticheta_corecta_departament = col1_fb.selectbox(
                        "Departament Corect:", options=options_departament,
                        index=list(options_departament).index(eticheta_departament)  # Pre-selectează
                    )

                    eticheta_corecta_urgenta = col2_fb.selectbox(
                        "Urgență Corectă:", options=options_urgenta,
                        index=list(options_urgenta).index(eticheta_urgenta)  # Pre-selectează
                    )

                    # Butonul de trimitere a formularului
                    submitted = st.form_submit_button("Trimite Corecția")

                    if submitted:
                        # --- Logica de salvare în CSV ---
                        try:
                            # 1. Creăm rândul nou de date
                            new_data_row = {
                                'data_raport': datetime.date.today(),
                                'text_raport': text_nou_utilizator,  # Textul original introdus
                                'eticheta_problema': eticheta_corecta_problema,
                                'eticheta_locatie': eticheta_corecta_locatie,
                                'eticheta_urgenta': eticheta_corecta_urgenta,
                                'eticheta_departament': eticheta_corecta_departament
                            }

                            new_df = pd.DataFrame([new_data_row])

                            # 2. Adăugăm rândul la fișierul CSV
                            new_df.to_csv(
                                "rapoarte_mentenanta_v2.csv",
                                mode='a',  # 'a' = append (adaugă)
                                header=False,  # Nu mai scrie header-ul
                                index=False,
                                encoding='utf-8-sig'
                            )

                            # 3. Informăm utilizatorul și curățăm cache-ul
                            st.success("Mulțumim pentru feedback! Corecția a fost salvată.")
                            st.balloons()

                            # Curățăm cache-ul pentru ca pagina "Analiză" să se actualizeze
                            st.cache_data.clear()

                        except Exception as e:
                            st.error(f"A apărut o eroare la salvarea feedback-ului: {e}")
            # --- Sfârșitul blocului "Human-in-the-Loop" ---

# --- PAGINA 2: ANALIZĂ (Dashboard-ul Managerial) ---
elif page == "Dashboard Analiză":

    st.title("📊 Dashboard Managerial - Analiză Mentenanță")

    if df_analiza is not None:
        st.subheader(f"Analiza celor {len(df_analiza)} rapoarte colectate")

        # ... (Tot codul tău pentru grafice rămâne neschimbat) ...
        # Grafic 1: Alocare pe Departament
        st.markdown("---")
        st.subheader("1. Încărcare pe Departament")
        st.markdown("Acest grafic arată câte tichete a primit fiecare departament.")
        distributie_dep = df_analiza['eticheta_departament'].value_counts()
        st.bar_chart(distributie_dep)

        # Grafic 2: Cele mai frecvente probleme
        st.markdown("---")
        st.subheader("2. Cele mai frecvente tipuri de defecțiuni")
        distributie_prob = df_analiza['eticheta_problema'].value_counts().head(10)
        st.bar_chart(distributie_prob)

        # Grafic 3: Distribuția Urgențelor
        st.markdown("---")
        st.subheader("3. Distribuția Urgențelor")
        st.markdown("Câte probleme au fost critice vs. minore?")
        distributie_urg = df_analiza['eticheta_urgenta'].value_counts()
        st.bar_chart(distributie_urg)  # Folosim bar_chart, așa cum am corectat

        # Bonus: Afișarea datelor brute
        st.markdown("---")
        st.subheader("Vizualizare Date Brute")
        if st.checkbox("Arată datele brute structurate"):
            st.dataframe(df_analiza)

    else:
        st.error("Nu s-au putut încărca datele pentru analiză. Verifică fișierul 'rapoarte_mentenanta_v2.csv'.")