from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load your trained model
with open("XGBoost_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load and preprocess data (same logic as Streamlit version)
url = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/refs/heads/master/data/2025-26/gws/merged_gw.csv"
df = pd.read_csv(url, on_bad_lines='skip', engine='python')
df.columns = [c.lower().replace(" ", "_") for c in df.columns]

base_features = [
    'minutes', 'goals_scored', 'assists', 'clean_sheets', 'goals_conceded',
    'own_goals', 'penalties_saved', 'penalties_missed', 'yellow_cards',
    'red_cards', 'saves', 'bonus', 'bps', 'influence', 'creativity',
    'threat', 'ict_index', 'value'
]

df = df.dropna(subset=base_features + ['total_points'])
df = df.sort_values(by=['name', 'round'])

# Precompute rolling features
players = []
for player in df['name'].unique():
    player_df = df[df['name'] == player].sort_values('round')
    if len(player_df) < 5:
        continue
    last5 = player_df.tail(5)
    avg_features = {f'avg_{f}': last5[f].mean() for f in base_features}
    avg_features.update({
        'name': player,
        'team': last5['team'].values[-1],
        'avg_points_last_5_gws': round(last5['total_points'].mean(), 2),
        'current_round': int(last5['round'].values[-1])
    })
    players.append(avg_features)

pred_df = pd.DataFrame(players)
X_test = pred_df[[f'avg_{f}' for f in base_features]]
X_test.columns = [c.replace('avg_', '') for c in X_test.columns]

@app.route('/')
def index():
    player_names = sorted(pred_df['name'].unique())
    return render_template('index.html', players=player_names)

@app.route('/predict', methods=['POST'])
def predict():
    player_name = request.form['player']
    player_data = pred_df[pred_df['name'] == player_name]
    X_player = X_test.loc[player_data.index]
    prediction = model.predict(X_player)[0]
    team = player_data['team'].values[0]
    avg_points = player_data['avg_points_last_5_gws'].values[0]
    return render_template('index.html', players=sorted(pred_df['name'].unique()),
                           prediction=round(prediction, 2),
                           player_name=player_name,
                           team=team,
                           avg_points=avg_points)

if __name__ == "__main__":
    app.run(debug=True)
