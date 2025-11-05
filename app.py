from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------
MODEL_FILE = "XGBoost_model.pkl"  # Change to your pickle file name
DATA_URL = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/refs/heads/master/data/2025-26/gws/merged_gw.csv"

BASE_FEATURES = [
    'minutes', 'goals_scored', 'assists', 'clean_sheets', 'goals_conceded',
    'own_goals', 'penalties_saved', 'penalties_missed', 'yellow_cards',
    'red_cards', 'saves', 'bonus', 'bps', 'influence', 'creativity',
    'threat', 'ict_index', 'value'
]

# ------------------------------------------------------------
# APP SETUP
# ------------------------------------------------------------
app = Flask(__name__)

# Load trained model
try:
    with open(MODEL_FILE, "rb") as f:
        model = pickle.load(f)
    print(f"✅ Model loaded from {MODEL_FILE}")
except Exception as e:
    raise RuntimeError(f"❌ Could not load model: {e}")

# Load and preprocess player data
print("📊 Loading FPL dataset...")
df = pd.read_csv(DATA_URL, on_bad_lines='skip', engine='python')
df.columns = [c.lower().replace(" ", "_") for c in df.columns]

# Add missing name if split
if 'name' not in df.columns and {'first_name', 'second_name'}.issubset(df.columns):
    df['name'] = df['first_name'] + " " + df['second_name']

# Ensure columns exist
missing = set(['name', 'team', 'round', 'total_points']) - set(df.columns)
if missing:
    raise RuntimeError(f"❌ Missing columns: {missing}")

df = df.dropna(subset=BASE_FEATURES + ['total_points']).sort_values(['name', 'round'])

# Build rolling 5-GW dataset
players = []
for player in df['name'].unique():
    p_df = df[df['name'] == player].sort_values('round')
    if len(p_df) < 5:
        continue
    last5 = p_df.tail(5)
    entry = {f'avg_{f}': last5[f].mean() for f in BASE_FEATURES}
    entry.update({
        'name': player,
        'team': last5['team'].iloc[-1],
        'avg_points_last_5_gws': last5['total_points'].mean(),
        'current_round': int(last5['round'].iloc[-1])
    })
    players.append(entry)

pred_df = pd.DataFrame(players)
X_test = pred_df[[f'avg_{f}' for f in BASE_FEATURES]]
X_test.columns = BASE_FEATURES
name_to_idx = {n: i for i, n in enumerate(pred_df['name'])}

# ------------------------------------------------------------
# ROUTES
# ------------------------------------------------------------
@app.route("/", methods=["GET"])
def home():
    players = sorted(pred_df['name'].unique())
    return render_template("index.html", players=players)

@app.route("/predict", methods=["POST"])
def predict():
    player_name = request.form['player']
    if player_name not in name_to_idx:
        return jsonify({"error": "Player not found"}), 404

    idx = name_to_idx[player_name]
    X_player = X_test.iloc[[idx]]
    y_pred = model.predict(X_player)[0]
    row = pred_df.iloc[idx]

    return render_template(
        "index.html",
        players=sorted(pred_df['name'].unique()),
        prediction=round(float(y_pred), 2),
        player=player_name,
        team=row['team'],
        avg_points=round(float(row['avg_points_last_5_gws']), 2)
    )

# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)
