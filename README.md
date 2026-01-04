# Cricket Win Prediction App 🏏

A **Neo-Brutalist** styled Cricket Win Prediction application built with **Streamlit** and **Machine Learning**.

## Features
- **🔮 Pre-Match Prediction**: Predict winner based on teams, venue, and toss.
- **🔴 Live In-Play Prediction**: Real-time win probability based on current score, wickets, and overs.
- **⚡ Neo-Brutalist UI**: High contrast, bold typography, and Simpson-themed aesthetics.
- **🧠 Machine Learning**: Uses a Random Forest Classifier trained on synthetic historical data.

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/dhananjayaDev/cricket-predictions-mini.git
   cd cricket-predictions-mini
   ```

2. **Set up Virtual Environment**
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # Mac/Linux
   source venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the App**
   ```bash
   streamlit run app.py
   ```

## Configuration
- **Background Image**: Replace `simp1.jpeg` in the root directory to change the background.
- **API Key**: Enter your [CricketData](https://cricketdata.org/) API key in the sidebar for live data features (optional).

## Tech Stack
- Python 3.11+
- Streamlit
- Scikit-learn
- Pandas & NumPy

## License
MIT
