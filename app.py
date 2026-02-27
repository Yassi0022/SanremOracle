# app.py — COMPLETO (Sanremo Predictor v2: Spotify live + refresh predictions)
from flask import Flask, render_template, jsonify
import os
import csv
import warnings
import numpy as np
import pandas as pd
import joblib

from spotipy import Spotify
from spotipy.oauth2 import SpotifyClientCredentials

warnings.filterwarnings("ignore")
app = Flask(__name__)

# =========================
# Config
# =========================
DATA_2026_PATH = "data/sanremo_2026.csv"
PRED_PATH = "data/predictions.csv"
MODEL_PATH = "model.pkl"
SCALER_PATH = "scaler.pkl"

FEATURES = ["quote_inv", "spotify_norm", "ig_norm", "Televoto_Proxy", "Stampa_Proxy", "Radio_Proxy"]


# =========================
# Helpers: Spotify client
# =========================
def get_spotify_client() -> Spotify:
    
    spotify_id = os.getenv("SPOTIFY_ID")
    spotify_secret = os.getenv("SPOTIFY_SECRET")

    if not spotify_id or not spotify_secret:
        raise RuntimeError("Missing SPOTIFY_ID / SPOTIFY_SECRET env vars")

    auth_manager = SpotifyClientCredentials(
        client_id=spotify_id,
        client_secret=spotify_secret
    )
    return Spotify(auth_manager=auth_manager)


# =========================
# Load model/scaler (una volta)
# =========================
def load_model_and_scaler():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Missing {MODEL_PATH}. Esegui prima: python model.py")
    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(f"Missing {SCALER_PATH}. Esegui prima: python model.py")

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler


model, scaler = load_model_and_scaler()


# =========================
# Feature engineering
# =========================
def prepare_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Quote inverse
    df["quote_inv"] = 1 / df["Quote"].replace(0, np.nan).fillna(30)

    # Normalizzazioni
    for col, norm in [("Spotify_ML", "spotify_norm"), ("IG_Followers", "ig_norm")]:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        mn, mx = df[col].min(), df[col].max()
        df[norm] = (df[col] - mn) / (mx - mn) if mx > mn else 0.5

    # Proxy (televoto/stampa/radio)
    for c in ["Televoto_Proxy", "Stampa_Proxy", "Radio_Proxy"]:
        if c not in df.columns:
            df[c] = 0.5
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.5)

    return df[FEATURES].fillna(0)


def load_predictions() -> list[dict]:
    data: list[dict] = []
    if not os.path.exists(PRED_PATH):
        return data

    with open(PRED_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if "Prob_Vittoria" in row and row["Prob_Vittoria"] not in (None, ""):
                row["Prob_Vittoria"] = float(row["Prob_Vittoria"])
            data.append(row)
    return data


def refresh_predictions() -> pd.DataFrame:
    """
    Ricalcola predictions.csv a partire dal sanremo_2026.csv corrente
    usando model.pkl + scaler.pkl
    """
    df_2026 = pd.read_csv(DATA_2026_PATH, encoding="utf-8")

    X = prepare_df(df_2026)
    Xs = scaler.transform(X)

    predicted_scores = model.predict(Xs)

    exp_scores = np.exp(predicted_scores / 3)
    probs = (exp_scores / np.sum(exp_scores)) * 100

    df_2026["Prob_Vittoria"] = np.round(probs, 1)
    df_sorted = df_2026.sort_values("Prob_Vittoria", ascending=False).reset_index(drop=True)
    df_sorted["Rank"] = df_sorted.index + 1

    df_sorted.to_csv(PRED_PATH, index=False, encoding="utf-8")
    return df_sorted


# =========================
# Spotify live update
# =========================
def spotify_popularity_for_row(sp: Spotify, artista: str, canzone: str) -> int | None:
    """
    Cerca la traccia su Spotify. Se non la trova, restituisce None
    così non sovrascriviamo il valore buono nel CSV con uno zero.
    """
    # 1° tentativo: ricerca stretta
    q_strict = f"track:{canzone} artist:{artista}"
    res = sp.search(q=q_strict, type="track", limit=1)
    items = res.get("tracks", {}).get("items", [])
    
    if items:
        return int(items[0].get("popularity", 0))

    # 2° tentativo: ricerca larga (solo parole chiave)
    q_loose = f"{canzone} {artista}"
    res_loose = sp.search(q=q_loose, type="track", limit=1)
    items_loose = res_loose.get("tracks", {}).get("items", [])
    
    if items_loose:
        return int(items_loose[0].get("popularity", 0))

    # Se proprio non trova nulla, torna None
    return None

# =========================
# Routes
# =========================
@app.route("/")
def home():
    pred = load_predictions()

    # Se non c'è ancora predictions.csv, calcolalo al volo
    if not pred:
        try:
            refresh_predictions()
            pred = load_predictions()
        except Exception:
            pred = []

    top15 = pred[:15]
    return render_template("index.html", top15=top15)


@app.route("/api/top")
def api_top():
    pred = load_predictions()
    top10 = [
        {"Artista": row.get("Artista"), "Prob_Vittoria": row.get("Prob_Vittoria")}
        for row in pred[:10]
    ]
    return jsonify(top10)


@app.route("/refresh", methods=["GET"])
def refresh():
    df_sorted = refresh_predictions()
    return jsonify({"status": "ok", "rows": int(df_sorted.shape[0]), "top1": df_sorted.iloc[0]["Artista"]})


# =========================
# Spotify live update
# =========================
def spotify_popularity_for_row(sp: Spotify, artista: str, canzone: str) -> int | None:
    """
    Cerca la traccia su Spotify. Se non la trova, restituisce None
    così non sovrascriviamo il valore buono nel CSV con uno zero.
    """
    # 1° tentativo: ricerca stretta
    q_strict = f"track:{canzone} artist:{artista}"
    res = sp.search(q=q_strict, type="track", limit=1)
    items = res.get("tracks", {}).get("items", [])
    
    if items:
        return int(items[0].get("popularity", 0))

    # 2° tentativo: ricerca larga (solo parole chiave)
    q_loose = f"{canzone} {artista}"
    res_loose = sp.search(q=q_loose, type="track", limit=1)
    items_loose = res_loose.get("tracks", {}).get("items", [])
    
    if items_loose:
        return int(items_loose[0].get("popularity", 0))

    # Se proprio non trova nulla, torna None
    return None

# =========================
# Routes
# =========================
# ... tieni le route home, api_top, refresh uguali ...

@app.route("/live_spotify", methods=["GET"])
def live_spotify():
    """
    Aggiorna Spotify_ML nel CSV e ricalcola predictions.csv.
    """
    try:
        sp = get_spotify_client()
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

    df = pd.read_csv(DATA_2026_PATH, encoding="utf-8")

    if "Artista" not in df.columns or "Canzone" not in df.columns:
        return jsonify({"status": "error", "message": "CSV must contain columns: Artista, Canzone"}), 400

    if "Spotify_ML" not in df.columns:
        df["Spotify_ML"] = 0

    updated = 0
    not_found = []

    for i, row in df.iterrows():
        artista = str(row["Artista"])
        canzone = str(row["Canzone"])
        
        # Pulizia base dei nomi (es. togliere feat. dal nome artista per aiutare la ricerca)
        art_search = artista.split("&")[0].strip() # Cerca solo il primo artista se c'è un duo
        
        pop = spotify_popularity_for_row(sp, art_search, canzone)
        
        if pop is not None and pop > 0:
            df.loc[i, "Spotify_ML"] = pop
            updated += 1
        else:
            not_found.append(artista)
            # NON sovrascriviamo con 0. Lasciamo il valore vecchio (presunto buono)

    df.to_csv(DATA_2026_PATH, index=False, encoding="utf-8")
    df_sorted = refresh_predictions()

    return jsonify({
        "status": "ok",
        "spotify_updated_rows": updated,
        "not_found": not_found, # Così vedi chi non ha trovato
        "top1": df_sorted.iloc[0]["Artista"],
        "top1_prob": float(df_sorted.iloc[0]["Prob_Vittoria"]),
    })


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=True)
