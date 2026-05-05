# 🎵 SanremOracle

**SanremOracle** is a machine learning project that predicts the final ranking of the Sanremo Music Festival (Italy's biggest annual song contest) using a regression model trained on multi-source data.

Built for the 2026 edition, it combines bookmaker odds, streaming metrics, and social signals to generate a probability ranking for all 30 competing artists.

***

## 🧠 How It Works

The model treats the prediction as a **regression problem**: instead of a binary win/loss classification, it learns to predict a continuous score (position from 1 to 30) for each artist. This avoids the "cliff effect" of classification and produces smooth, ranked probabilities.

### Feature Engineering

| Feature | Source | Description |
|---|---|---|
| `quote_inv` | Bookmaker odds | Inverse of betting odds — the strongest single predictor |
| `spotify_norm` | Spotify API (live) | Normalized monthly listeners at time of prediction |
| `ig_norm` | Manual collection | Normalized Instagram follower count |
| `Televoto_Proxy` | Estimated | Proxy score for predicted televote performance |
| `Stampa_Proxy` | Estimated | Proxy for press/media coverage |
| `Radio_Proxy` | Estimated | Proxy for radio airplay |

### Model

- **Algorithm**: `RandomForestRegressor` (scikit-learn)
- **Estimators**: 500 trees, max depth 5
- **Target**: `score_target = 31 - Posizione` (position 1 → score 30, position 30 → score 1)
- **Scaling**: `StandardScaler` applied before training
- **Output conversion**: Scores are exponentiated with a softening factor (`/ 3`) and normalized via softmax to produce percentage win probabilities

```
exp_scores = np.exp(predicted_scores / 3)
probs = (exp_scores / sum(exp_scores)) * 100
```

### Training Data

Historical Sanremo results (`hist_sanremo.csv`) augmented with synthetic data for positions 6–30 using realistic monotonic interpolation across all features, to give the model a full position curve to learn from.

***

## 🚀 Architecture

```
model.py        → trains and saves model.pkl + scaler.pkl
app.py          → Flask server with live Spotify refresh endpoint
templates/      → Frontend (Jinja2 HTML)
data/
  hist_sanremo.csv    → historical results (training set)
  sanremo_2026.csv    → 2026 artist data
  predictions.csv     → model output
```

The Flask app exposes a `/refresh` endpoint that fetches **live Spotify data** via the Spotipy client, re-runs inference, and returns updated rankings — so predictions evolve as streaming numbers change during the festival week.

***

## 📦 Tech Stack

- **Python 3.11**
- **scikit-learn** — RandomForestRegressor, StandardScaler
- **Flask** — web server and API
- **Spotipy** — Spotify Web API client (live data refresh)
- **pandas / numpy** — data processing
- **joblib** — model serialization

***

## ⚙️ Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set environment variables

```bash
export SPOTIFY_ID=your_spotify_client_id
export SPOTIFY_SECRET=your_spotify_client_secret
```

### 3. Train the model

```bash
python model.py
```

This generates `model.pkl`, `scaler.pkl`, and `data/predictions.csv`.

### 4. Run the app

```bash
python app.py
```

Open `http://localhost:5000` to see the ranked predictions.

***

## 📊 Sample Output (2026 Predictions)

```
 1. Lucio Corsi              | 12.4% | ██████
 2. Olly                     | 10.8% | █████
 3. Giorgia                  |  9.1% | ████
 4. Fedez                    |  7.3% | ███
 5. Brunori Sas              |  6.2% | ███
...
```

***

## 💡 Key Design Decisions

- **Regression over classification**: Predicting a continuous score rather than binary win/loss produces more nuanced, differentiated probability distributions.
- **Bookmaker odds as anchor feature**: `quote_inv` is intentionally weighted as the dominant signal — historical data confirms it's the best single predictor of festival outcomes.
- **Live data refresh**: Spotify streaming numbers change daily during festival week; the `/refresh` endpoint allows re-inference without retraining.

***

## 📁 Related Projects

- [HobbyBuddy](https://github.com/Yassi0022/hobbybuddy) — Social matching platform (Spring Boot + FastAPI + React)
- [HR-Attrition-Analysis](https://github.com/Yassi0022/HR-Attrition-Analysis) — Employee attrition analysis with IBM dataset

***

## 👤 Author

**Yassine Hatouf** · [GitHub](https://github.com/Yassi0022) · [LinkedIn](https://linkedin.com/in/yassine-hatouf)
