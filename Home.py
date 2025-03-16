import streamlit as st
import os

# ✅ CONFIGURACIÓN DE LA PÁGINA
st.set_page_config(
    layout="wide",
    page_title="Automated Trading System"
)

# ✅ CARGAR CSS EXTERNO
with open("style.css") as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# ✅ TÍTULO PRINCIPAL
st.markdown('<p class="dashboard_title" style="text-align: center; font-size: 50px; font-weight: bold; margin-bottom: 30px;">🚀 Automated Trading System</p>', unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 3, 1])

with col2:
    # Definir estilos en HTML para los botones
    button_style = """
    <style>
    .nav-button {
        display: inline-block;
        padding: 12px 20px;
        font-size: 35px; 
        font-weight: bold;
        text-align: center;
        border-radius: 8px;
        border: 2px solid #f7931a;
        color: white;
        background-color: #1a1a1a;
        cursor: pointer;
        transition: 0.3s;
        width: 200px;
        margin: 5px;
    }
    .nav-button:hover {
        background-color: #f7931a;
        color: black;
    }
    </style>
    """

    st.markdown(button_style, unsafe_allow_html=True)

    # Crear 3 columnas para los botones
    col_b1, col_b2, col_b3 = st.columns([1, 1, 1])

    selected_page = None

    with col_b1:
        if st.button("📄 Overview", key="overview"):
            selected_page = "📄 Overview"

    with col_b2:
        if st.button("📡 Core Features", key="features"):
            selected_page = "📡 Core Features"

    with col_b3:
        if st.button("👥 Meet the Team", key="team"):
            selected_page = "👥 Meet the Team"

    # Si no se ha seleccionado nada, mostrar la primera opción
    if not selected_page:
        selected_page = "📄 Overview"

# -------------------------------------
# ✅ CONTENIDO DINÁMICO SEGÚN LA SECCIÓN SELECCIONADA
# -------------------------------------
if selected_page == "📄 Overview":
    st.markdown("## 📄 Overview")  # Keeps the section title

    st.markdown(
    """
        - #### Welcome to the <strong>Automated Daily Trading System</strong>, an advanced Python-based platform designed to automate daily stock trading.
        - #### This system leverages cutting-edge <strong>machine learning algorithms</strong> to predict market movements and provides an <strong>interactive web-based interface</strong> for traders to monitor and execute strategies in real-time.

    """,
    unsafe_allow_html=True
)


elif selected_page == "📡 Core Features":
    st.markdown("## 📡Core Features")
    st.markdown(
        """
        - #### **Data Analytics Module:** Develops a machine learning model for market movement forecasting based on historical data from at least five major US companies.
        - #### **Web-Based Application:** A user-friendly multi-page interactive dashboard built with Streamlit to analyze stock trends and interact with predictive models.
        - #### **Live Trading Page:** Real-time stock movement visualization and execution of trades for selected stock tickers.
        - #### **Trading Strategy Module:** Implementation of machine learning-based trading strategies to optimize investment decisions.
        """
    )
elif selected_page == "👥 Meet the Team":
    st.markdown("<h2 style='text-align: center;'>👥 Meet the Team</h2>", unsafe_allow_html=True)

    # ✅ DATOS DEL EQUIPO
    IMAGE_DIR = "team/"  # Folder where images are stored

    team_members = [
        {"name": "Massimo Tassinari", "role": "Lead Model Creation", "image": os.path.join(IMAGE_DIR, "massimo.jpeg")},
        {"name": "Pablo Viaña", "role": "Lead API Configuration", "image": os.path.join(IMAGE_DIR, "pablo.jpeg")},
        {"name": "Maha Alkaabi", "role": "Lead Streamlit", "image": os.path.join(IMAGE_DIR, "maha.jpeg")},
        {"name": "Yotaro Enomoto", "role": "Lead Streamlit", "image": os.path.join(IMAGE_DIR, "yotaro.JPG")},
        {"name": "Yihang Li", "role": "Lead Model Creation", "image": os.path.join(IMAGE_DIR, "yihang.jpeg")}
    ]


    # ✅ CREAR COLUMNAS IGUALES PARA LOS MIEMBROS DEL EQUIPO
    team_cols = st.columns(len(team_members))

    for i, member in enumerate(team_members):
        with team_cols[i]:
            image_path = member["image"]

            # ✅ **Use st.image() to display images correctly**
            if os.path.exists(image_path):
                st.image(image_path, width=140)
            else:
                st.warning(f"⚠️ Image not found: {image_path}")

            # ✅ Display name & role with improved styling
            st.markdown(f"<p style='text-align: center; font-size: 18px; font-weight: bold; margin-top: 5px;'>{member['name']}</p>", unsafe_allow_html=True)
            st.markdown(f"<p style='text-align: center; font-size: 16px; color: gray;'>{member['role']}</p>", unsafe_allow_html=True)