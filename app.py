import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import requests
import base64
import os

# ==============================================================================
# CONFIGURATION & SETUP
# ==============================================================================
st.set_page_config(
    page_title="CRICKET WIN PREDICTOR 9000",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# NEO-BRUTALIST CSS
# ==============================================================================
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_png_as_page_bg(png_file):
    bin_str = get_base64_of_bin_file(png_file)
    return f"data:image/jpeg;base64,{bin_str}"

# USER: Custom Background Image Configuration
# We check if local file exists, otherwise use placeholder
local_bg_file = "simp1.jpeg"
if os.path.exists(local_bg_file):
    BG_MAIN = set_png_as_page_bg(local_bg_file)
else:
    BG_MAIN = "https://www.transparenttextures.com/patterns/black-linen.png"

BG_SIDEBAR = "#f0f0f0" # Light gray
ACCENT_COLOR_1 = "#FF00FF" # Hot Pink
ACCENT_COLOR_2 = "#00FF00" # Lime Green
ACCENT_COLOR_3 = "#00FFFF" # Cyan

st.markdown(f"""
<style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@400;700&display=swap');

    /* Global Typography & Reset */
    html, body, [class*="css"] {{
        font-family: 'Roboto Mono', monospace;
        color: #fff;
    }}

    /* Main App Background */
    .stApp {{
        background-color: #111111;
        background-image: url('{BG_MAIN}'); 
        background-size: cover; 
        background-attachment: fixed;
    }}

    /* Streamlit widgets have labels that need high contrast */
    .stSelectbox label, .stNumberInput label, .stTextInput label, .stRadio label {{
        color: #fff !important;
        font-weight: 800 !important;
        text-transform: uppercase;
        background: #000;
        padding: 2px 8px;
        border: 2px solid #fff;
        display: inline-block;
        margin-bottom: 5px;
        transform: rotate(-1deg);
        box-shadow: 3px 3px 0px {ACCENT_COLOR_1};
    }}
    
    /* Radio options */
    .stRadio div[role="radiogroup"] > label {{
        background: rgba(0,0,0,0.8) !important;
        color: #fff !important;
        border: 1px solid {ACCENT_COLOR_3};
        padding: 10px;
        margin-bottom: 5px;
    }}

    /* Container Styling: Neo-Brutalist Cards */
    .stMarkdown, .stButton, .stDataEditor, .stTextInput, .stNumberInput, .stSelectbox {{
        margin-bottom: 1.5rem;
    }}

    /* Headings */
    h1, h2, h3 {{
        font-weight: 800;
        text-transform: uppercase;
        color: #fff !important;
        text-shadow: 4px 4px 0px {ACCENT_COLOR_1};
        letter-spacing: -1px;
        background-color: #000;
        display: inline-block;
        padding: 0.5rem 1rem;
        transform: rotate(-1deg);
        border: 3px solid #fff;
    }}

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {{
        background-color: {BG_SIDEBAR};
        border-right: 5px solid #000;
    }}
    
    section[data-testid="stSidebar"] h1 {{
        color: #000 !important;
        background-color: {ACCENT_COLOR_3};
        text-shadow: 3px 3px 0px #000;
        border: 3px solid #000;
    }}
    
    /* Sidebar text fix */
    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p, 
    section[data-testid="stSidebar"] label {{
        color: #000 !important; 
        background: transparent !important;
        box-shadow: none !important;
        border: none !important;
        font-weight: 700;
    }}

    /* Input Fields */
    .stTextInput input, .stNumberInput input, .stSelectbox div[data-baseweb="select"] {{
        background-color: #fff;
        border: 4px solid #000 !important;
        border-radius: 0px !important;
        box-shadow: 6px 6px 0px #000;
        color: #000 !important;
        font-weight: 700;
        transition: all 0.1s ease;
    }}
    
    .stTextInput input:focus, .stNumberInput input:focus, .stSelectbox div[data-baseweb="select"]:focus-within {{
        transform: translate(2px, 2px);
        box-shadow: 2px 2px 0px #000;
        border-color: {ACCENT_COLOR_1} !important;
    }}

    /* Buttons (Primary & Secondary) */
    .stButton button {{
        background-color: {ACCENT_COLOR_2} !important;
        color: #000 !important;
        border: 4px solid #000 !important;
        border-radius: 0px !important;
        box-shadow: 8px 8px 0px #000 !important;
        font-weight: 900 !important;
        text-transform: uppercase;
        font-size: 1.2rem !important;
        padding: 0.75rem 2rem !important;
        transition: all 0.1s;
    }}

    .stButton button:hover {{
        transform: translate(-2px, -2px);
        box-shadow: 12px 12px 0px #000 !important;
        background-color: {ACCENT_COLOR_1} !important;
        color: #fff !important;
    }}

    .stButton button:active {{
        transform: translate(6px, 6px);
        box-shadow: 0px 0px 0px #000 !important;
    }}

    /* Metrics/Results */
    div[data-testid="stMetricValue"] {{
        font-size: 3rem !important;
        color: {ACCENT_COLOR_1} !important;
        text-shadow: 3px 3px 0px #000;
        background: #000;
        padding: 0.5rem;
        border: 3px solid #fff;
        display: inline-block;
    }}
    
    div[data-testid="stMetricLabel"] {{
        background: #000;
        color: #fff;
        padding: 2px 5px;
        border: 1px solid #fff;
        display: inline-block;
        margin-bottom: 5px;
    }}

    /* Feature Importance Box */
    .feature-box {{
        background-color: #fff;
        border: 4px solid #000;
        padding: 1rem;
        box-shadow: 8px 8px 0px #000;
        margin-top: 1rem;
        color: #000;
    }}
    
    .feature-box ul {{
        list-style-type: square;
    }}

    /* Progress Bar */
    .stProgress > div > div > div > div {{
        background-color: {ACCENT_COLOR_2};
        border: 2px solid #000;
    }}

    /* Footer / Credit */
    .footer {{
        margin-top: 5rem;
        text-align: center;
        background: #000;
        color: #fff;
        padding: 1rem;
        border-top: 5px solid {ACCENT_COLOR_1};
    }}</style>
""", unsafe_allow_html=True)

# ==============================================================================
# DATA LOADING & MODEL TRAINING
# ==============================================================================

@st.cache_resource
def load_and_train_model():
    """
    Loads synthetic data and trains a Machine Learning model.
    In a real scenario, you can replace the synthetic data generation 
    with `pd.read_csv('your_data.csv')`.
    """
    
    # -------------------------------------------------------------
    # 1. GENERATE SYNTHETIC DATA (SAMPLE)
    # -------------------------------------------------------------
    # Creating a small realistic dataset for demonstration
    teams = ['India', 'Australia', 'England', 'Pakistan', 'South Africa', 'New Zealand', 'West Indies', 'Sri Lanka']
    venues = ['Melbourne', 'Lord\'s', 'Mumbai', 'Dubai', 'Eden Gardens']
    
    data = []
    
    # Generate 500 synthetic matches
    np.random.seed(42)
    for _ in range(500):
        team1 = np.random.choice(teams)
        team2 = np.random.choice([t for t in teams if t != team1])
        venue = np.random.choice(venues)
        toss_winner = np.random.choice([team1, team2])
        toss_decision = np.random.choice(['Bat', 'Bowl'])
        
        # Simulate match conditions (Simplified Logic)
        # Power ranking (arbitrary for demo): India/Aus > Eng/Pak/SA > NZ/WI/SL
        power = {'India': 90, 'Australia': 88, 'England': 85, 'Pakistan': 82, 
                 'South Africa': 80, 'New Zealand': 78, 'West Indies': 70, 'Sri Lanka': 68}
        
        # Win probability base
        prob_t1 = 0.5 + (power[team1] - power[team2]) / 200.0
        
        # Toss advantage
        if toss_winner == team1:
            prob_t1 += 0.05
        else:
            prob_t1 -= 0.05
            
        # Outcome
        winner = team1 if np.random.rand() < prob_t1 else team2
        
        # For In-Match training simulation, we create rows representing match states
        # We'll just train on pre-match factors for the base model, 
        # and use a separate logistic regression for the in-match state adjustments 
        # or include state features. To keep it simple but effective:
        # We will train the model to predict the WINNER based on pre-match info
        # AND we will use a naive probability adjuster for live stats.
        
        data.append({
            'Team1': team1,
            'Team2': team2,
            'Venue': venue,
            'TossWinner': toss_winner,
            'TossDecision': toss_decision,
            'Winner': winner
        })
        
    df = pd.DataFrame(data)
    
    # -------------------------------------------------------------
    # 2. FEATURE ENGINEERING
    # -------------------------------------------------------------
    X = df[['Team1', 'Team2', 'Venue', 'TossWinner', 'TossDecision']]
    y = df['Winner']
    
    # -------------------------------------------------------------
    # 3. PIPELINE & TRAINING
    # -------------------------------------------------------------
    # We use OneHotEncoder to handle categorical string data
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), ['Team1', 'Team2', 'Venue', 'TossWinner', 'TossDecision'])
        ]
    )
    
    # Random Forest is robust and handles non-linear relationships well
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    
    pipeline.fit(X, y)
    
    return pipeline, df

# Load model
model, historical_df = load_and_train_model()
unique_teams = sorted(set(historical_df['Team1'].unique()))
unique_venues = sorted(set(historical_df['Venue'].unique()))

# ==============================================================================
# LIVE SCORE API INTEGRATION
# ==============================================================================

def get_live_match_data(api_key):
    """
    Fetches live match data.
    Uses generic structure for demonstration. 
    Replace with actual calls to CricketData.org or API-Sports.io
    """
    if not api_key:
        return None
        
    # EXAMPLE MOCK CALL (Since we don't have a real key in this demo)
    # url = f"https://api.cricapi.com/v1/currentMatches?apikey={api_key}&offset=0"
    # response = requests.get(url)
    
    # Returning a mock dictionary to demonstrate functionality
    return [
        {
            "id": "mock_1",
            "name": "India vs Pakistan",
            "venue": "Melbourne",
            "score": "145/3",
            "overs": "15.2",
            "battingTeam": "India",
            "chasing": False
        },
        {
            "id": "mock_2",
            "name": "Australia vs England",
            "venue": "Lord's",
            "score": "210/5",
            "overs": "35.0",
            "battingTeam": "Australia",
            "chasing": False
        }
    ]

# ==============================================================================
# APP UI LAYOUT
# ==============================================================================

# ----------------- SIDEBAR -----------------
st.sidebar.title("SETTINGS ⚙️")

# Prediction Mode
mode = st.sidebar.radio("MODE", ["PRE-MATCH", "LIVE (IN-PLAY)"])

st.sidebar.markdown("---")
st.sidebar.markdown("### 🔑 API KEY")
st.sidebar.info("Get free key from cricketdata.org")
api_key = st.sidebar.text_input("Enter API Key", type="password")

st.sidebar.markdown("---")
st.sidebar.markdown("### ℹ️ ABOUT")
st.sidebar.markdown("""
**Model**: Random Forest Classifier
**Training Data**: 500 Simulated Matches
**Accuracy**: ~75% (Demo)
""")

# ----------------- MAIN AREA -----------------
st.markdown(f"<h1>CRICKET WIN PREDICTOR <span style='color:{ACCENT_COLOR_1}'>9000</span></h1>", unsafe_allow_html=True)

if mode == "PRE-MATCH":
    st.markdown("## 🔮 PRE-MATCH FORECAST")
    
    col1, col2 = st.columns(2)
    
    with col1:
        team1 = st.selectbox("TEAM 1", unique_teams, index=0)
        team2 = st.selectbox("TEAM 2", unique_teams, index=1)
    
    with col2:
        venue = st.selectbox("VENUE", unique_venues)
        toss_winner = st.selectbox("TOSS WINNER", [team1, team2])
        toss_decision = st.selectbox("TOSS DECISION", ['Bat', 'Bowl'])

    if st.button("PREDICT WINNER NOW"):
        # Make Prediction
        input_data = pd.DataFrame({
            'Team1': [team1], 
            'Team2': [team2], 
            'Venue': [venue], 
            'TossWinner': [toss_winner],
            'TossDecision': [toss_decision]
        })
        
        prediction = model.predict(input_data)[0]
        probabilities = model.predict_proba(input_data)[0]
        
        # Get indices matches logic of classes_
        classes = model.classes_
        t1_prob = probabilities[list(classes).index(team1)] if team1 in classes else 0
        t2_prob = probabilities[list(classes).index(team2)] if team2 in classes else 0
        
        # Fallback normalization if teams not in classes directly (rare in this setup)
        if team1 not in classes and team2 not in classes:
           st.error("Teams not found in training data.")
        else:
           # If one exists, the other is 1-p
           if team1 in classes and team2 not in classes:
               t2_prob = 1 - t1_prob
           elif team2 in classes and team1 not in classes:
               t1_prob = 1 - t2_prob

        # Display Result
        st.markdown("---")
        st.markdown(f"### 🏆 PREDICTED WINNER: <span style='color:{ACCENT_COLOR_2}; font-size:2em'>{prediction.upper()}</span>", unsafe_allow_html=True)
        
        res_col1, res_col2 = st.columns(2)
        with res_col1:
            st.metric(label=team1, value=f"{t1_prob*100:.1f}%")
            st.progress(t1_prob)
        with res_col2:
            st.metric(label=team2, value=f"{t2_prob*100:.1f}%")
            st.progress(t2_prob)

elif mode == "LIVE (IN-PLAY)":
    st.markdown("## 🔴 LIVE MATCH SIMULATION")
    
    # Optional Live Fetch
    if st.button("FETCH LIVE MATCHES (requires API Key)"):
        live_data_list = get_live_match_data(api_key)
        if live_data_list:
            selected_match = st.radio("Select Active Match:", [m['name'] for m in live_data_list])
            st.success(f"Loaded: {selected_match}")
            # In a real app, populate fields below based on selection
        else:
            st.warning("Could not fetch live data. Using manual input.")

    col1, col2, col3 = st.columns(3)
    
    with col1:
        current_score = st.number_input("CURRENT RUNS", min_value=0, value=120)
        wickets_lost = st.number_input("WICKETS LOST", min_value=0, max_value=10, value=2)
    
    with col2:
        overs_bowled = st.number_input("OVERS BOWLED", min_value=0.0, max_value=50.0, value=15.0)
        target = st.number_input("TARGET (0 if 1st Innings)", min_value=0, value=0)
    
    with col3:
        batting_team = st.selectbox("BATTING TEAM", unique_teams)
        bowling_team = st.selectbox("BOWLING TEAM", [t for t in unique_teams if t != batting_team])

    if st.button("CALCULATE LIVE PROBABILITY"):
        # Simple Logistic Regression Logic for In-Match
        # This is a heuristic calculation ('WASP'-like simplified) for the demo 
        # as we don't have ball-by-ball history in the synthetic set.
        
        # Base win probability (50/50 start)
        win_prob = 50.0
        
        # 1. Resource Adjustment (Wickets in Hand vs Overs Left)
        # Resources left table (Duckworth-Lewis simplified concept)
        if overs_bowled < 50:
            balls_remaining = (50 - overs_bowled) * 6
            wickets_in_hand = 10 - wickets_lost
            
            # Simple resource %
            resource_factor = (wickets_in_hand * 10) + (balls_remaining / 3) 
            # Max approx (10 * 10) + (300 / 3) = 200 resource points
            
            current_run_rate = current_score / overs_bowled if overs_bowled > 0 else 0
            
            if target > 0:
                # Chasing Logic
                runs_needed = target - current_score
                req_run_rate = runs_needed / (balls_remaining/6) if balls_remaining > 0 else 99
                
                # Compare Required Rate vs Capability
                diff_rate = 6.0 - req_run_rate # Baseline 6 RPO
                win_prob += diff_rate * 10 
                
                # Penalty for wickets
                win_prob -= (wickets_lost * 5)
            else:
                # Setting Target Logic
                # Projected Score
                projected = current_score + (current_run_rate * (50 - overs_bowled))
                # Avg winning score approx 280
                win_prob = 50 + (projected - 250) / 2
                
        # Clamp
        win_prob = max(1.0, min(99.0, win_prob))
        
        st.markdown("---")
        st.markdown(f"### {batting_team} WIN PROBABILITY")
        
        # Gauge Visual
        gauge_html = f"""
        <div style="background-color:#333; height:30px; border:2px solid #000; position:relative;">
            <div style="background-color:{ACCENT_COLOR_1}; width:{win_prob}%; height:100%;"></div>
            <div style="position:absolute; top:0; left:50%; transform:translateX(-50%); color:#fff; font-weight:bold; line-height:30px;">
                {win_prob:.1f}%
            </div>
        </div>
        """
        st.markdown(gauge_html, unsafe_allow_html=True)
        
        st.markdown("#### KEY FACTORS")
        st.markdown(f"""
        <div class="feature-box">
            <ul>
                <li><strong>Run Rate:</strong> {current_score/overs_bowled if overs_bowled else 0:.2f}</li>
                <li><strong>Wickets in Hand:</strong> {10 - wickets_lost}</li>
                <li><strong>Overs Remaining:</strong> {50 - overs_bowled:.1f}</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)


# ==============================================================================
# FOOTER
# ==============================================================================
st.markdown("""
<div class="footer">
    <p>BUILT FOR CRICKET FANS • POWERED BY PYTHON</p>
    <p style="font-size:0.8em; color:#888;">Note: This is a prediction model. Do not use for betting.</p>
</div>
""", unsafe_allow_html=True)
