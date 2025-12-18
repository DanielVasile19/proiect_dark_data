import os
import time  # <--- NOU: Pentru benchmark latenta
import datetime
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
import io  # <--- NOU: Pentru buffer memorie la model summary

#UI
st.set_page_config(
    page_title="SIA Dispecerat Industrial",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS
st.markdown("""
    <style>
    .big-font { font-size:18px !important; }
    .stMetric { background-color: #0e1117; border: 1px solid #303030; padding: 10px; border-radius: 5px; }
    .success-time { color: #00ff00; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

#Căi
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR = os.path.join(BASE_DIR, 'models')


def get_valid_data_path():
    possible_paths = [
        os.path.join(BASE_DIR, 'data', 'generated', 'raw', 'rapoarte_mentenanta_v2.csv'),
        os.path.join(BASE_DIR, 'data', 'raw', 'rapoarte_mentenanta_v2.csv')
    ]
    for path in possible_paths:
        if os.path.exists(path):
            return path
    return possible_paths[0]


DATA_PATH = get_valid_data_path()

def get_urgency_color(urgency_text):
    if urgency_text.lower() == 'critica':
        return 'inverse'
    elif urgency_text.lower() == 'medie':
        return 'off'
    return 'normal'


@st.cache_resource
def load_resources():
    try:
        model_path = os.path.join(MODELS_DIR, 'trained_model.h5')
        if not os.path.exists(model_path):
            return None, None, None, None, None

        model = tf.keras.models.load_model(model_path)
        vect = joblib.load(os.path.join(MODELS_DIR, 'vectorizer_v2.joblib'))
        enc_p = joblib.load(os.path.join(MODELS_DIR, 'encoder_problema_v2.joblib'))
        enc_d = joblib.load(os.path.join(MODELS_DIR, 'encoder_departament_v2.joblib'))
        enc_u = joblib.load(os.path.join(MODELS_DIR, 'encoder_urgenta_v2.joblib'))
        return model, vect, enc_p, enc_d, enc_u
    except Exception as e:
        st.error(f"System Error: {e}")
        return None, None, None, None, None


def load_data():
    if os.path.exists(DATA_PATH):
        try:
            df = pd.read_csv(DATA_PATH)
            if 'data_raport' in df.columns:
                df['data_raport'] = pd.to_datetime(df['data_raport'], format='mixed', errors='coerce')
            return df
        except Exception as e:
            st.error(f"Eroare citire CSV: {e}")
            return pd.DataFrame()
    return pd.DataFrame()


# Initializare
models = load_resources()
df_history = load_data()

#Sidebar
st.sidebar.title("Meniu Principal")
page = st.sidebar.radio("Navigare", ["📝 Dispecerat Live", "📊 Dashboard Analitic", "⚙️ Detalii Tehnice Model"], index=0)
st.sidebar.divider()

# Afisare stare sistem in sidebar
if models[0]:
    st.sidebar.success("✅ Model RN Încărcat")
else:
    st.sidebar.error("❌ Model RN Lipsă")

st.sidebar.info(f"Database: {len(df_history)} înregistrări")

if page == "📝 Dispecerat Live":
    st.title("🏭 Dispecerat Mentenanță AI")

    col_left, col_right = st.columns([1, 1.2])

    with col_left:
        st.subheader("1. Raportare Incident")
        example = st.selectbox(
            "Exemple Rapide:",
            ["", "fum negru de la motorul 3, este urgent", "plc ars langa pompa, s-a oprit productia.",
             "s-a blocat pompa la robotul 5." , ]
        )
        input_val = example if example else ""
        text_input = st.text_area("Descriere:", value=input_val, height=150)
        btn_analyze = st.button("🔍 Analizează Raport", type="primary", use_container_width=True)

    if 'result' not in st.session_state:
        st.session_state['result'] = None

    if btn_analyze and text_input and models[0]:
        model, vec, ep, ed, eu = models
        with st.spinner("Inferență..."):
            # --- BENCHMARK START ---
            start_time = time.time()

            vec_in = vec.transform([text_input]).toarray()
            pred = model.predict(vec_in, verbose=0)

            end_time = time.time()
            # --- BENCHMARK END ---

            latency_ms = (end_time - start_time) * 1000

            st.session_state['result'] = {
                'text': text_input,
                'prob': ep.inverse_transform([np.argmax(pred[0])])[0],
                'dep': ed.inverse_transform([np.argmax(pred[1])])[0],
                'urg': eu.inverse_transform([np.argmax(pred[2])])[0],
                'conf_scores': [np.max(pred[0]), np.max(pred[1]), np.max(pred[2])],
                'latency': latency_ms
            }

    with col_right:
        st.subheader("2. Rezultat Analiză AI")
        res = st.session_state['result']

        if res:
            st.markdown(f"⏱️ Timp Inferență: <span class='success-time'>{res['latency']:.2f} ms</span>",
                        unsafe_allow_html=True)

            if res['latency'] < 50:
                st.caption("🚀 Performanță optimă (<50ms) - Real-Time Ready")

            min_conf = min(res['conf_scores'])
            if min_conf < 0.70:
                st.warning(f"Incertitudine ({min_conf:.1%}). Verifică manual!")
            else:
                st.success("Identificare sigură.")

            c1, c2, c3 = st.columns(3)
            c1.metric("Problemă", res['prob'].title(), f"{res['conf_scores'][0]:.0%}")
            c2.metric("Departament", res['dep'], f"{res['conf_scores'][1]:.0%}")
            c3.metric("Urgență", res['urg'].upper(), f"{res['conf_scores'][2]:.0%}",
                      delta_color=get_urgency_color(res['urg']))

            st.divider()

            with st.expander("🛠️ Corecție Manuală"):
                with st.form("feedback"):
                    _, _, ep, ed, eu = models
                    s_p = st.selectbox("Problemă", ep.classes_, index=list(ep.classes_).index(res['prob']))
                    s_d = st.selectbox("Departament", ed.classes_, index=list(ed.classes_).index(res['dep']))
                    s_u = st.selectbox("Urgență", eu.classes_, index=list(eu.classes_).index(res['urg']))

                    if st.form_submit_button("Salvează Feedback"):
                        new_row = {
                            'data_raport': datetime.datetime.now(),
                            'text_raport': res['text'],
                            'eticheta_problema': s_p, 'eticheta_departament': s_d, 'eticheta_urgenta': s_u,
                            'eticheta_locatie': 'Feedback_UI'
                        }
                        pd.DataFrame([new_row]).to_csv(DATA_PATH, mode='a', header=False, index=False)
                        st.toast("Salvat!", icon="💾")
        else:
            st.info("Așteptare input...")

#Dashboard
elif page == "📊 Dashboard Analitic":
    st.title("📊 Statistici Operaționale")

    if not df_history.empty:
        # Filtre
        all_deps = ['Toate'] + list(df_history['eticheta_departament'].unique())
        selected_dep = st.selectbox("Filtru Departament:", all_deps)

        df_filtered = df_history.copy()
        if selected_dep != 'Toate':
            df_filtered = df_filtered[df_filtered['eticheta_departament'] == selected_dep]

        # KPI
        k1, k2, k3 = st.columns(3)
        k1.metric("Total Tichete", len(df_filtered))
        k2.metric("Top Problemă",
                  df_filtered['eticheta_problema'].mode()[0].replace("_", " ") if not df_filtered.empty else "-")
        k3.metric("Critice", len(df_filtered[df_filtered['eticheta_urgenta'] == 'critica']))

        st.divider()

        # Grafice
        c1, c2 = st.columns(2)
        with c1:
            st.bar_chart(df_filtered['eticheta_problema'].value_counts())
        with c2:
            st.bar_chart(df_filtered['eticheta_urgenta'].value_counts(), color="#ff4b4b")

        #Exportul în CSV
        st.subheader("📋 Registru și Export")

        col_table, col_export = st.columns([3, 1])
        with col_table:
            st.dataframe(df_filtered.sort_index(ascending=False).head(50), use_container_width=True)

        with col_export:
            csv = df_filtered.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Descarcă CSV",
                data=csv,
                file_name='raport_mentenanta.csv',
                mime='text/csv',
                type="primary"
            )
    else:
        st.warning("Lipsă date.")

elif page == "⚙️ Detalii Tehnice Model":
    st.title("⚙️ Arhitectura Rețelei Neuronale")

    if models[0]:
        model = models[0]

        st.markdown("### Rezumat Model (Keras Summary)")

        stream = io.StringIO()
        model.summary(print_fn=lambda x: stream.write(x + '\n'))
        summary_string = stream.getvalue()

        st.code(summary_string, language='text')

        st.markdown("### Configurație Optimizator")
        st.json(model.optimizer.get_config())

    else:
        st.error("Modelul nu este încărcat.")