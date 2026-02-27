import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
# from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import joblib
import warnings
warnings.filterwarnings("ignore")

print("=" * 50)
print("  SANREMO 2026 PREDICTOR - MODELLO AVANZATO (Regressione)")
print("=" * 50)

FEATURES = ["quote_inv", "spotify_norm", "ig_norm", "Televoto_Proxy", "Stampa_Proxy", "Radio_Proxy"]

def prepare_df(df):
    df = df.copy()
    # Pesi reali: la quota dei bookmaker è il predittore migliore
    df["quote_inv"] = 1 / df["Quote"].replace(0, np.nan).fillna(30)
    
    # Normalizzazione lineare
    for col, norm in [("Spotify_ML", "spotify_norm"), ("IG_Followers", "ig_norm")]:
        mn, mx = df[col].min(), df[col].max()
        df[norm] = (df[col] - mn) / (mx - mn) if mx > mn else 0.5
        
    for c in ["Televoto_Proxy", "Stampa_Proxy", "Radio_Proxy"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.5)
        
    return df[FEATURES].fillna(0)

# 1. Carica storici
print("1. Caricamento dati storici e costruzione dataset...")
df_hist = pd.read_csv("data/hist_sanremo.csv", encoding="utf-8")

# Invece di vincitore o perdente (1 o 0), usiamo la Posizione!
# Più è vicina a 1, meglio è.
df_hist["score_target"] = 31 - df_hist["Posizione"] # Il 1° ha score 30, il 30° ha score 1

# --- CREIAMO TUTTO IL RESTO DELLA CLASSIFICA STORICA ---
# Riempiamo le posizioni da 6 a 30 per avere una curva realistica
posizioni_mancanti = np.arange(6, 31)
n_mancanti = len(posizioni_mancanti)

np.random.seed(100)
df_resto = pd.DataFrame({
    "Posizione": posizioni_mancanti,
    "Quote": np.linspace(15.0, 100.0, n_mancanti), # Le quote si alzano scendendo in classifica
    "Spotify_ML": np.linspace(800000, 50000, n_mancanti), # Gli ascolti scendono
    "IG_Followers": np.linspace(500000, 10000, n_mancanti),
    "Televoto_Proxy": np.linspace(0.6, 0.1, n_mancanti),
    "Stampa_Proxy": np.linspace(0.6, 0.1, n_mancanti),
    "Radio_Proxy": np.linspace(0.6, 0.1, n_mancanti)
})
df_resto["score_target"] = 31 - df_resto["Posizione"]
df_hist = pd.concat([df_hist, df_resto], ignore_index=True)
# --------------------------------------------------------

X_hist = prepare_df(df_hist)
y_hist = df_hist["score_target"] # Obiettivo continuo (1-30)

scaler = StandardScaler()
X_hist_s = scaler.fit_transform(X_hist)

print("2. Addestramento RandomForestRegressor...")
# La regressione evita l'effetto "muro" e crea probabilità morbide
model = RandomForestRegressor(
    n_estimators=500,
    max_depth=5,
    random_state=42
)
model.fit(X_hist_s, y_hist)

print("\n3. Calcolo predizioni sfumate per i 30 Big di Sanremo 2026...")
df_2026 = pd.read_csv("data/sanremo_2026.csv", encoding="utf-8")
X_2026 = prepare_df(df_2026)
X_2026_s = scaler.transform(X_2026)

# Prediciamo lo "Score" (da 1 a 30)
predicted_scores = model.predict(X_2026_s)

# Trasformiamo lo score in una percentuale realistica (Softmax / Normalizzazione)
# Esponenzializziamo per dare risalto ai primi 3 e staccarli dal gruppo (come avviene nel televoto reale)
exp_scores = np.exp(predicted_scores / 3) # Il diviso 3 ammorbidisce la curva
probs = (exp_scores / np.sum(exp_scores)) * 100

df_2026["Prob_Vittoria"] = np.round(probs, 1)
df_sorted = df_2026.sort_values("Prob_Vittoria", ascending=False).reset_index(drop=True)
df_sorted["Rank"] = df_sorted.index + 1

print("\nTOP 15 PREDIZIONI SANREMO 2026 (Realistico):")
print("-" * 55)
for _, row in df_sorted.head(15).iterrows():
    bar = "█" * int(row["Prob_Vittoria"] / 2) # Diviso 2 per non farle lunghissime
    if bar == "": bar = "▏"
    print(f"{int(row['Rank']):2}. {row['Artista']:25} | {row['Prob_Vittoria']:4.1f}% | {bar}")
print("-" * 55)

df_sorted.to_csv("data/predictions.csv", index=False)
joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("\nModello di regressione salvato!")