import os
import datetime
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf

# Configurare pagina
st.set_page_config(
    page_title="Dispecer AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configurare cai
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')
RAW_DIR = os.path.join(BASE_DIR, 'data', 'raw')
DATA_PATH = os.path.join(RAW_DIR, 'rapoarte_mentenanta_v2.csv')


@st.cache_resource
def load_models():
    """Incarca modelele ML si encoderele salvate."""
    try:
        model_path = os.path.join(PROCESSED_DIR, 'model_dispecer_v2.keras')
        if not os.path.exists(model_path):
            st.error("Modelul nu a fost gasit. Asigurati-va ca ati rulat antrenarea.")
            return None, None, None, None, None

        model = tf.keras.models.load_model(model_path)
        vectorizer = joblib.load(os.path.join(PROCESSED_DIR, 'vectorizer_v2.joblib'))
        enc_prob = joblib.load(os.path.join(PROCESSED_DIR, 'encoder_problema_v2.joblib'))
        enc_dep = joblib.load(os.path.join(PROCESSED_DIR, 'encoder_departament_v2.joblib'))
        enc_urg = joblib.load(os.path.join(PROCESSED_DIR, 'encoder_urgenta_v2.joblib'))

        return model, vectorizer, enc_prob, enc_dep, enc_urg
    except Exception as e:
        st.error(f"Eroare critica la incarcarea modelelor: {e}")
        return None, None, None, None, None


@st.cache_data
def load_data(file_path):
    """Incarca datele istorice din CSV."""
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    return pd.DataFrame()


# Initializare resurse
models = load_models()
df_history = load_data(DATA_PATH)

# Bara laterala
st.sidebar.title("Navigatie")
page = st.sidebar.radio("Meniu:", ["Dispecerat Inteligent", "Dashboard Analiza"])
st.sidebar.markdown("---")
st.sidebar.info("Sistem Inteligent de Dispecerat\nFIIR 2025")

# --- PAGINA 1: DISPECERAT ---
if page == "Dispecerat Inteligent":
    st.title("🤖 Dispecerat Mentenanta")
    st.markdown("### Analiza Automata a Rapoartelor de Defectiune")

    col_input, col_result = st.columns([1, 1])

    with col_input:
        st.subheader("📝 Raportare")
        text_input = st.text_area(
            "Descriere defectiune (text liber):",
            value="mtoor ars pe linia 1. Urgent, linia blocata.",
            height=150,
            help="Descrieti problema intampinata."
        )
        btn_analyze = st.button("Analizeaza Raport", type="primary")

    # Gestionare stare sesiune
    if 'analysis_result' not in st.session_state:
        st.session_state['analysis_result'] = None

    # Logica de predictie
    if btn_analyze and models[0]:
        model, vectorizer, enc_prob, enc_dep, enc_urg = models

        with st.spinner("Procesare AI..."):
            # Vectorizare
            text_vec = vectorizer.transform([text_input]).toarray()

            # Predictie
            pred = model.predict(text_vec, verbose=0)

            # Decodare rezultate
            idx_p = np.argmax(pred[0][0])
            idx_d = np.argmax(pred[1][0])
            idx_u = np.argmax(pred[2][0])

            st.session_state['analysis_result'] = {
                'text': text_input,
                'prob': enc_prob.inverse_transform([idx_p])[0],
                'conf_p': pred[0][0][idx_p],
                'dep': enc_dep.inverse_transform([idx_d])[0],
                'conf_d': pred[1][0][idx_d],
                'urg': enc_urg.inverse_transform([idx_u])[0],
                'conf_u': pred[2][0][idx_u]
            }

    # Afisare rezultate
    if st.session_state['analysis_result']:
        res = st.session_state['analysis_result']

        with col_result:
            st.subheader("📊 Rezultat AI")
            st.success("Analiza finalizata cu succes")

            c1, c2, c3 = st.columns(3)
            c1.metric("Problema", res['prob'].title(), f"{res['conf_p']:.0%}")
            c2.metric("Departament", res['dep'].title(), f"{res['conf_d']:.0%}")
            c3.metric("Urgenta", res['urg'].title(), f"{res['conf_u']:.0%}")

            st.divider()

            # Sectiune Feedback (Human-in-the-Loop)
            with st.expander("⚠️ Corectie Manuala (Feedback)"):
                with st.form("feedback_form"):
                    st.write("Corectati etichetele daca este necesar:")

                    _, _, enc_p, enc_d, enc_u = models

                    sel_p = st.selectbox("Problema Reala", enc_p.classes_,
                                         index=list(enc_p.classes_).index(res['prob']))
                    sel_d = st.selectbox("Departament Corect", enc_d.classes_,
                                         index=list(enc_d.classes_).index(res['dep']))
                    sel_u = st.selectbox("Urgenta Reala", enc_u.classes_, index=list(enc_u.classes_).index(res['urg']))

                    if st.form_submit_button("Salveaza Corectia"):
                        try:
                            new_entry = pd.DataFrame([{
                                'data_raport': datetime.date.today(),
                                'text_raport': res['text'],
                                'eticheta_problema': sel_p,
                                'eticheta_locatie': 'manual_feedback',
                                'eticheta_urgenta': sel_u,
                                'eticheta_departament': sel_d
                            }])

                            new_entry.to_csv(DATA_PATH, mode='a', header=False, index=False)
                            st.success("Feedback salvat! Sistemul va invata din acest exemplu.")
                            st.cache_data.clear()
                        except Exception as e:
                            st.error(f"Eroare la salvare: {e}")

# --- PAGINA 2: ANALIZA ---
elif page == "Dashboard Analiza":
    st.title("📈 Statistici Mentenanta")

    if not df_history.empty:
        # Reincarcare date pentru a include feedback-ul recent
        df_history = load_data(DATA_PATH)

        st.markdown(f"**Total rapoarte procesate:** `{len(df_history)}`")
        st.markdown("---")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Top Defectiuni")
            chart_data = df_history['eticheta_problema'].value_counts().head(7)
            st.bar_chart(chart_data)

        with col2:
            st.subheader("Incarcare pe Departamente")
            chart_data = df_history['eticheta_departament'].value_counts()
            st.bar_chart(chart_data)

        st.subheader("Distributie Urgenta")
        st.bar_chart(df_history['eticheta_urgenta'].value_counts())

        with st.expander("Vizualizare Date Brute"):
            st.dataframe(df_history)

    else:
        st.warning("Nu exista date istorice disponibile pentru analiza.")