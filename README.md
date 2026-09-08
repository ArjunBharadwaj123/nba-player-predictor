# 🏀🏈 Multi-Sport Player Predictor (NBA + NFL)

A machine learning-powered application that predicts a player's next-game
performance using advanced statistical modeling and real-world data pipelines.
It now covers **two sports** as separate pages behind one app and one API:

- **`/nba`** — the original NBA predictor (unchanged).
- **`/nfl`** — a full NFL predictor (QB/RB/WR/TE/K) built on **nflverse**
  (`nflreadpy`): position-specific projections, prediction intervals, PPR /
  Half-PPR / No-PPR fantasy points, football-language SHAP explanations, and
  over/under probabilities.
- **`/`** — redirects to `/nba`. A compact NBA/NFL switcher is on every page.

Built for fantasy players, sports bettors, and data enthusiasts, this tool
provides **stat predictions, confidence ranges, and probability insights** for
upcoming games. See `docs/TECHNICAL_OVERVIEW.md` (Part 2) for the full NFL
design.

---

## 🏈 NFL quick start

```bash
# 1. Install (adds nflreadpy / polars / pyarrow on top of the NBA deps)
pip install -r requirements.txt

# 2. Build data + train + evaluate.
#    Development mode (fewer seasons, fast, clearly labelled "dev"):
NFL_DEV_MODE=1 python -m nfl.pipeline.update
#    Full production run (2018 → latest; configurable, heavier):
python -m nfl.pipeline.update --seasons 2018,2019,2020,2021,2022,2023,2024

# 3. Serve (NBA at the root, NFL under /nfl — same app)
uvicorn api.main:app --reload

# 4. Frontend
cd dashboard && npm install && npm run dev   # visit /nfl
```

Useful NFL commands:

```bash
python -m nfl.scraping.collect            # download + cache seasons
python -m nfl.features.build_dataset      # inspect the canonical dataset
python -m nfl.models.train                # train per (position,target)
python -m nfl.models.evaluate             # chronological walk-forward report
python -m nfl.models.tune --position QB   # small chronological grid search
python -m nfl.pipeline.update --dry-run   # detect new games, write nothing
```

Tests:

```bash
pytest                       # Python: scoring, positions, leakage, API, NBA compat
cd dashboard && npm run lint && npm run build && npm test   # frontend + scoring persistence
```

The NFL system works **without** `ODDS_API_KEY` (odds are omitted and marked
missing, never faked). Dev-mode artifacts are labelled `"dev"` and surfaced as a
UI warning — never presented as production accuracy.

### 🔒 Admin dashboard & in-season updates

- **`/admin`** — a password-protected dashboard (NBA vs NFL train/test metrics +
  a combined confidence-weighted **Top-10**). Set the password server-side:
  `ADMIN_PASSWORD` env var on the API host (never commit it). The page prompts
  for it; the API checks it via the `X-Admin-Password` header.
- **Stays current in-season.** NFL models train on **2018–2025**, weighting the
  current season more, and only support **currently-rostered** players (retirees
  drop off; a player's current team — and its offensive environment — drive the
  projection, so movers are modeled in their new situation). A player with **no
  upcoming game gets no projection.**
- **Current teams & depth charts.** Each player's current team comes from
  `nflverse` `players.latest_team` (authoritative and correct in the offseason, so
  a trade like A.J. Brown → NE is reflected immediately), and starter status comes
  from the current-season **depth chart** (disambiguated by the current team, with
  a recent-snap-share fallback). Both are computed in the pipeline and stored in
  the committed `players.json`, so the deployed API needs no live fetch per request.
- **Weekly auto-retrain.** `.github/workflows/retrain.yml` runs Tuesdays (noon
  EST), refreshes + retrains both sports, rebuilds the Top-10, and commits the
  artifacts to `main` (→ Render + Vercel redeploy). Regenerate the Top-10
  manually with `python -m analytics.top_predictions`.

---

## 🚀 Features

- 🔮 Predicts next-game player stats:
  - Points, Rebounds, Assists
  - Steals, Blocks, Minutes
  - Fantasy Score

- 📊 Confidence intervals  
  - Predictions include a **±1 standard deviation range**

- 📈 Probability calculator  
  - Calculate the probability a player goes **over/under a stat line**

- 🧠 Explainable AI (SHAP)
  - Understand *why* a prediction was made
  - Feature-level contribution insights

- ⚡ Real-time data integration
  - Fetches latest player stats and upcoming matchups

---

## 🧠 Problem & Motivation

Predicting player performance in the NBA is complex due to:
- Game-to-game variance
- Opponent strength
- Player usage and role changes
- Schedule fatigue and rest days

This project aims to **quantify player performance probabilistically**, giving users a **data-driven edge** in fantasy sports and betting scenarios.

---

## 🏗️ Tech Stack

### Backend / ML
- Python
- XGBoost
- SHAP
- pandas, scikit-learn
- FastAPI
- BeautifulSoup

### Data Sources
- Basketball Reference (scraping)
- NBA API
- balldontlie API

### Frontend
- React (Vite)

### Visualization
- matplotlib

---

## 🧪 Model Details

- **Model:** XGBoost (Gradient Boosting)
- **Outputs:**
  - Points, rebounds, assists, steals, blocks, minutes, fantasy score

### Features Used
- Player stats:
  - Usage rate (USG%)
  - Rolling averages (recent performance)
  - Season averages
  - Performance vs specific opponents
  - Rest days / fatigue

- Team & context stats:
  - Team pace
  - Opponent pace
  - Defensive rating & rank
  - Opponent stats allowed by position

- Advanced engineered features:
  - Trend indicators
  - Efficiency metrics (TS%, points-per-minute)
  - Recency-weighted (EWMA) form
  - Home/away form splits
  - Matchup interactions (opponent pace, combined game pace, usage × minutes)
  - **Projected minutes fed into the counting-stat models** (minutes stacking)
  - Game context, schedule timing, starter status

---

## ⚙️ Pipeline

1. **Data Collection**
   - Player game logs scraped from Basketball Reference
   - Team stats from NBA API
   - Upcoming games via balldontlie API

2. **Feature Engineering**
   - True usage rate calculation  
   - Rolling averages & trends  
   - Opponent-specific performance  
   - Fatigue and schedule features  
   - Position-based defensive matchups  

3. **Model Training**
   - One XGBoost model per stat, tuned with 5-fold `TimeSeriesSplit` CV
   - Hyperparameters chosen from evidence via `models/tune.py` (MAE/R² sweep
     graphs in `reports/tuning/`) — shallow trees (`max_depth=3`) generalise
     best on noisy game-level data
   - Minutes is trained first; its out-of-fold projection is fed to the
     counting-stat models (leakage-free stacking)
   - Two extra quantile models per stat (p15 / p85) for calibrated intervals

4. **Prediction & Explainability**
   - SHAP used for per-prediction feature attribution and reasoning strings
   - Calibrated p15–p85 quantile prediction ranges (asymmetric, minutes-aware)
   - Warnings for load-management / back-to-back / thin data

---

## 📊 Results

Backtested on a **chronological holdout** (train on the earliest games, test on
the most recent 15% — never on the future) via `models/evaluate.py`:

| Stat    | MAE   | RMSE  | R²    | Interval coverage |
|---------|-------|-------|-------|-------------------|
| Points  | 5.44  | 6.98  | 0.50  | 68%               |
| Rebounds| 2.17  | 2.83  | 0.51  | 68%               |
| Assists | 1.66  | 2.20  | 0.52  | 69%               |
| Steals  | 0.78  | 0.99  | 0.07  | 82%               |
| Blocks  | 0.61  | 0.82  | 0.21  | 81%               |
| Minutes | 4.10  | 5.44  | 0.58  | 70%               |

- **Points / Rebounds / Assists / Minutes:** the most predictable stats; R² ≈ 0.5
  is strong for *game-level* NBA prediction, where game-to-game variance is high.
- **Steals / Blocks:** rare, near-random events — low R² is inherent, so the
  models correctly stay close to the mean and their intervals over-cover.
- **Prediction intervals** land near the 70% target for the main stats — the
  bands are calibrated, not cosmetic.

Metrics: MAE (Mean Absolute Error), RMSE (Root Mean Squared Error), R² Score, and
interval coverage. Re-generate anytime with `python nba/models/evaluate.py`.

---

## 🚀 Getting Started

### 1. Clone the repo

git clone https://github.com/YOUR_USERNAME/nba-player-predictor.git
cd nba-player-predictor

### 2. Backend Setup

pip install -r requirements.txt
uvicorn api.main:app --reload

### 3. Frontend Setup

cd dashboard
npm install
npm run dev

---

## 📁 Project Structure

The two sports are **symmetric, self-contained packages** — `nba/` and `nfl/`
mirror each other, and neither imports the other. `api/` mounts both; the
frontend has one page per sport.

```
├── README.md
├── api                          # FastAPI app (one server, both sports)
│   ├── main.py                  #   NBA endpoints at root  +  mounts NFL router
│   └── nfl.py                   #   NFL router, mounted under /nfl
│
├── nba                          # 🏀  NBA predictor (all NBA backend code)
│   ├── config.py                #   targets, fantasy weights, paths
│   ├── scraping/                #   bbref_scraper, nba_api_client, next_game, odds
│   ├── features/                #   build_dataset.py, engineer.py, feature_config.py
│   ├── models/                  #   train.py, tune.py, evaluate.py, saved/*.pkl
│   ├── explainability/          #   shap_explainer.py (predictions + SHAP + quantiles)
│   ├── pipeline/                #   update.py (in-season re-scrape + retrain)
│   ├── reports/                 #   hyperparameter sweep charts
│   └── data/                    #   raw/ (gitignored) + processed serving CSVs
│
├── nfl                          # 🏈  NFL predictor (all NFL backend code)
│   ├── config.py                #   seasons, positions, targets, normalization
│   ├── scoring.py               #   PPR/kicker scoring + fantasy simulation
│   ├── serving.py               #   serve-time feature-row builder
│   ├── scraping/                #   collect.py (nflreadpy+cache), odds, next-game
│   ├── features/                #   build_dataset.py, engineer.py, feature_config.py
│   ├── models/                  #   train.py, evaluate.py, tune.py, saved/<POS>/
│   ├── explainability/          #   explainer.py (football-language SHAP)
│   ├── pipeline/                #   update.py (idempotent, atomic)
│   └── data/                    #   raw/ cache/ (gitignored) + processed serving
│
├── dashboard                    # React (Vite) frontend
│   └── src
│       ├── pages/               #   NbaPage.jsx 🏀  ·  NflPage.jsx 🏈
│       ├── components/          #   shared UI (Card, StatBar, PlayerSearch, SportNav)
│       └── lib/                 #   api, nflConfig, fantasy, useLocalStorage hook
│
├── tests                        # pytest: scoring, positions, leakage, API, NBA compat
└── docs
    └── TECHNICAL_OVERVIEW.md     # full technical breakdown (Part 1 NBA, Part 2 NFL)
```

**Which code is which?** Anything under `nba/` (and `NbaPage.jsx`) is the NBA
predictor; anything under `nfl/`, `api/nfl.py`, and `NflPage.jsx` is the NFL
predictor. Files under `dashboard/src/components` and `dashboard/src/lib` are
shared by both pages.

---

## 🔍 Example Output

- Predicted stats with confidence ranges  
- SHAP-based reasoning:
  - "High scoring baseline (last 10 avg: 21.3 pts)"
  - "Opponent weak defensive rating"
- Probability calculation:
  - "63% chance of scoring over 20 points"

---

## 🖥️ Application Preview

![App Screenshot](./assets/screenshot.png)

---

## 💡 Future Improvements

- 🐳 Docker containerization  
- ☁️ Cloud deployment (AWS / GCP)  
- 🏥 Injury-aware modeling  
- 📡 Live updating predictions  
- 📊 Betting edge detection  

---

## ⚠️ Disclaimer

This tool is for educational and analytical purposes only.  
Predictions are probabilistic and should not be considered guaranteed outcomes.

---

## 👤 Author

Arjun Bharadwaj  
GitHub: https://github.com/ArjunBharadwaj123