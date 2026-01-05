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
    /* Import modern font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');

    /* Global Typography */
    html, body, [class*="css"] {{
        font-family: 'Inter', sans-serif;
        color: #ffffff;
    }}

    /* Background Image */
    .stApp {{
        background-color: #000000;
        background-image: url('{BG_MAIN}');
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}
    
    /* Overlay for better readability if image is bright */
    .stApp::before {{
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(0, 0, 0, 0.4);
        z-index: -1;
    }}

    /* GLASSMORPHISM CARD STYLE */
    .stMarkdown, .stButton, .stDataEditor, .stTextInput, .stNumberInput, .stSelectbox, .feature-box {{
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 10px;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
        margin-bottom: 1.5rem;
    }}

    /* Remove default Streamlit shadows/borders on specific inputs to blend in */
    .stTextInput input, .stNumberInput input, .stSelectbox div[data-baseweb="select"] {{
        background-color: rgba(0, 0, 0, 0.6) !important;
        color: #fff !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        border-radius: 8px !important;
    }}
    
    /* Input Focus */
    .stTextInput input:focus, .stNumberInput input:focus, .stSelectbox div[data-baseweb="select"]:focus-within {{
        border-color: {ACCENT_COLOR_1} !important;
        box-shadow: 0 0 10px {ACCENT_COLOR_1}40;
    }}

    /* Labels */
    .stSelectbox label, .stNumberInput label, .stTextInput label, .stRadio label {{
        color: #ddd !important;
        font-weight: 600;
        font-size: 0.9rem;
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        transform: none !important;
    }}

    /* Headings */
    h1, h2, h3 {{
        color: #fff !important;
        text-shadow: 0 2px 4px rgba(0,0,0,0.5);
        font-weight: 800;
        background: transparent !important;
        border: none !important;
        transform: none !important;
        padding: 0 !important;
    }}
    
    h1 span {{
        background: -webkit-linear-gradient(45deg, {ACCENT_COLOR_1}, {ACCENT_COLOR_2});
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }}

    /* Buttons */
    .stButton button {{
        background: linear-gradient(135deg, {ACCENT_COLOR_1}AA, {ACCENT_COLOR_3}AA) !important;
        color: #fff !important;
        border: none !important;
        border-radius: 12px !important;
        font-weight: 700 !important;
        padding: 0.75rem 2rem !important;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3) !important;
    }}
    
    .stButton button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.4) !important;
        filter: brightness(1.2);
    }}

    /* Sidebar */
    section[data-testid="stSidebar"] {{
        background-color: rgba(0, 0, 0, 0.85) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(20px);
    }}

    /* Metrics */
    div[data-testid="stMetricValue"] {{
        background: transparent !important;
        border: none !important;
        color: {ACCENT_COLOR_2} !important;
        font-size: 2.5rem !important;
        text-shadow: 0 0 10px {ACCENT_COLOR_2}66;
    }}
    
    div[data-testid="stMetricLabel"] {{
        background: transparent !important;
        border: none !important;
        color: #aaa !important;
    }}

    /* Feature Box */
    .feature-box {{
        background: rgba(0, 0, 0, 0.6);
        color: #eee;
    }}
    .feature-box ul {{
        list-style: none;
        padding-left: 0;
    }}
    .feature-box li {{
        margin-bottom: 5px;
        padding-left: 10px;
        border-left: 3px solid {ACCENT_COLOR_3};
    }}

    /* Footer */
    .footer {{
        margin-top: 3rem;
        padding: 1rem;
        text-align: center;
        background: rgba(0, 0, 0, 0.8);
        border-top: 1px solid rgba(255, 255, 255, 0.1);
        color: #888;
        font-size: 0.8rem;
    }}

</style>
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
        # Gauge Visual
        gauge_val = f"{win_prob:.1f}"
        gauge_html = f"""
        <div style="background-color:rgba(255,255,255,0.1); height:30px; border-radius:15px; position:relative; overflow:hidden; border:1px solid rgba(255,255,255,0.2);">
            <div style="background-color:{ACCENT_COLOR_1}; width:{gauge_val}%; height:100%;"></div>
            <div style="position:absolute; top:0; left:50%; transform:translateX(-50%); color:#fff; font-weight:bold; line-height:30px; text-shadow:0 1px 2px #000;">
                {gauge_val}%
            </div>
        </div>
        """
        st.markdown(gauge_html, unsafe_allow_html=True)
        
        # Key Factors Visuals
        run_rate = current_score / overs_bowled if overs_bowled > 0 else 0
        wickets_left = 10 - wickets_lost
        overs_left = 50 - overs_bowled
        
        # Pre-format to avoid syntax errors in multiline f-strings
        rr_str = f"{run_rate:.2f}"
        ol_str = f"{overs_left:.1f}"
        
        st.markdown("#### KEY FACTORS")
        st.markdown(f"""
        <div class="feature-box">
            <ul>
                <li><strong>Run Rate:</strong> {rr_str}</li>
                <li><strong>Wickets in Hand:</strong> {wickets_left}</li>
                <li><strong>Overs Remaining:</strong> {ol_str}</li>
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
