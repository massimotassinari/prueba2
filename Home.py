import streamlit as st
import os

# ✅ Must be the first command
st.set_page_config(
    layout="wide",
    page_title="Automated Trading System"
)

# Load external CSS for styling
def load_css():
    st.markdown(
        """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&display=swap');
            
            /* REMOVE STREAMLIT DEFAULT SPACING */
            .main .block-container {
                padding: 0 !important;
                margin: 0 !important;
                max-width: 100% !important;
            }

            /* HIDE FOOTER & HEADER */
            footer {display: none !important;}
            [data-testid="stHeader"] {display: none !important;}

            /* GLOBAL FONT & COLOR SETTINGS */
            html, body, [class*="css"] {
                font-family: 'Space Grotesk', sans-serif !important;
                background-color: #0f0f0f !important;
                color: #f6f6f6 !important;
            }

            /* PAGE TITLE */
            .dashboard_title {
                font-size: 100px; 
                font-family: 'Space Grotesk';
                font-weight: 700;
                line-height: 1.2;
                text-align: left;
            }

            /* NAVIGATION CONTAINER */
            .nav-container {
                display: flex;
                justify-content: center;
                background-color: #111;
                padding: 15px 30px;
                width: 100%;
                border-bottom: 1px solid #333;
            }

            /* NAVIGATION LINKS */
            .nav-text {
                font-size: 18px;
                font-weight: bold;
                color: white;
                text-decoration: none;
                cursor: pointer;
                transition: color 0.3s ease;
            }
            .nav-text:hover {
                color: #f7931a;
            }

            /* CUSTOM MARKDOWN CLASSES */
            .stock_details {
                font-size: 30px; 
                font-family: 'Space Grotesk';
                color: #f6f6f6;
                font-weight: 900;
                text-align: left;
                line-height: 1;
                padding-bottom: 10px;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

load_css()

# ---------------------------
# LAYOUT: TITLE AND NAVIGATION
# ---------------------------
col1, col2, col3 = st.columns([1, 3, 1])  # Centering the radio more

with col1:
    st.markdown('<p class="dashboard_title">Automated <br>Trading <br>System</p>', unsafe_allow_html=True)

with col2:
    page_selection = st.radio(
        label="",
        options=["Overview", "Core Features", "Meet the Team"],
        horizontal=True,
        key="main_nav"
    )

# ---------------------------
# DYNAMIC CONTENT BASED ON SELECTION
# ---------------------------
if page_selection == "Overview":
    st.markdown("## 📄Overview")
    st.markdown(
        """
        - Welcome to the **Automated Daily Trading System**, an advanced Python-based platform designed to automate daily stock trading.
        - This system leverages cutting-edge machine learning algorithms to predict market movements and provides an interactive web-based interface for traders to monitor and execute strategies in real-time. 
        """
    )

elif page_selection == "Core Features":
    st.markdown("## 📡Core Features")
    st.markdown(
        """
        - **Data Analytics Module:** Develops a machine learning model for market movement forecasting based on historical data from at least five major US companies.
        - **Web-Based Application:** A user-friendly multi-page interactive dashboard built with Streamlit to analyze stock trends and interact with predictive models.
        - **Live Trading Page:** Real-time stock movement visualization and execution of trades for selected stock tickers.
        - **Trading Strategy Module:** Implementation of machine learning-based trading strategies to optimize investment decisions.
        """
    )

elif page_selection == "Meet the Team":
    st.markdown("## 👫Meet the Team")

    IMAGE_DIR = "team/"  # Folder where images are stored

    team_members = [
        {"name": "Massimo Tassinari", "role": "Lead Model Creation", "image": os.path.join(IMAGE_DIR, "massimo.jpeg")},
        {"name": "Pablo Viaña", "role": "Lead API Configuration", "image": os.path.join(IMAGE_DIR, "pablo.jpeg")},
        {"name": "Maha Alkaabi", "role": "Lead Streamlit", "image": os.path.join(IMAGE_DIR, "maha.jpeg")},
        {"name": "Yotaro Enomoto", "role": "Lead Streamlit", "image": os.path.join(IMAGE_DIR, "yotaro.JPG")},
        {"name": "Yihang Li", "role": "Lead Model Creation", "image": os.path.join(IMAGE_DIR, "yihang.jpeg")}
    ]

    team_cols = st.columns(len(team_members))  # Creates equal columns for members

    for i, member in enumerate(team_members):
        with team_cols[i]:
            if os.path.exists(member["image"]):
                st.image(member["image"], width=140)  # Load images correctly
            else:
                st.warning(f"Image not found: {member['image']}")  # Debugging for missing images
            
            st.markdown(f"<p style='text-align: center;'><b>{member['name']}</b></p>", unsafe_allow_html=True)
            st.markdown(f"<p style='text-align: center;'>{member['role']}</p>", unsafe_allow_html=True)
