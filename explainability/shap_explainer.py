"""
Step 15 & 16: SHAP Explainability
===================================
Generates SHAP values for each model prediction and converts them
into human-readable reasoning strings.

The output looks like this:

    Predicted Stats:
    Points: 27  Rebounds: 8  Assists: 6

    Reasoning:
    + Rolling form: averaging 31.2 pts over last 5 games
    + Weak opponent defense (rated 116.4, rank 28/30)
    + Home game advantage
    - Back-to-back game (fatigue factor)
    - Below-season scoring average recently

Usage:
    from explainability.shap_explainer import explain_prediction
    reasoning = explain_prediction(player_row, models, feature_names)

Or run standalone to verify on a sample row:
    python explainability/shap_explainer.py
"""

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import shap

ROOT       = Path(__file__).parent.parent
PROCESSED  = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models" / "saved"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

TARGETS = ["pts", "reb", "ast", "stl", "blk", "minutes"]

# Fantasy score weights
FANTASY_WEIGHTS = {
    "pts": 1.0, "reb": 1.2, "ast": 1.5,
    "stl": 3.0, "blk": 3.0,
}


# ─────────────────────────────────────────────────────────────────────────────
# Load models and features
# ─────────────────────────────────────────────────────────────────────────────

def load_models() -> dict:
    """Load all saved XGBoost models from disk."""
    models = {}
    for target in TARGETS:
        path = MODELS_DIR / f"{target}_model.pkl"
        if path.exists():
            with open(path, "rb") as f:
                models[target] = pickle.load(f)
            log.info(f"  Loaded {target}_model.pkl")
        else:
            log.warning(f"  Model not found: {path}")
    return models


def load_feature_names() -> list:
    """Load the canonical feature name list."""
    path = PROCESSED / "feature_names.txt"
    names = path.read_text().splitlines()
    # Add minutes variance feature if present in saved models
    if "rolling_last5_minutes_std" not in names:
        names.append("rolling_last5_minutes_std")
    return names


# ─────────────────────────────────────────────────────────────────────────────
# SHAP value computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_shap_values(model, X_row: pd.DataFrame) -> np.ndarray:
    """
    Compute SHAP values for a single prediction row.

    We use TreeExplainer — the fast, exact SHAP algorithm for tree-based
    models. For XGBoost it runs in O(TLD) time where T=trees, L=leaves,
    D=depth. For our models (~400 trees, depth 4) this takes ~50ms per row.

    Returns:
        1D numpy array of SHAP values, one per feature.
        Positive value  = feature pushed prediction UP
        Negative value  = feature pushed prediction DOWN
        Sum of all SHAP values + base_value = final prediction
    """
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X_row)

    # shap_values() returns shape (n_rows, n_features)
    # We always pass one row, so take [0]
    if len(shap_vals.shape) == 2:
        return shap_vals[0]
    return shap_vals


def get_base_value(model) -> float:
    """Get the model's base prediction (average prediction on training data)."""
    explainer = shap.TreeExplainer(model)
    return float(explainer.expected_value)


# ─────────────────────────────────────────────────────────────────────────────
# Human-readable reasoning
# ─────────────────────────────────────────────────────────────────────────────

# Maps feature name patterns to human-readable descriptions
# Each entry: (pattern_to_match, positive_label, negative_label)
FEATURE_LABELS = [
    # Rolling form
    ("rolling_last5_pts",      "Strong recent scoring (last 5 avg: {val:.1f} pts)",
                               "Cold scoring recently (last 5 avg: {val:.1f} pts)"),
    ("rolling_last10_pts",     "High scoring baseline (last 10 avg: {val:.1f} pts)",
                               "Low scoring baseline (last 10 avg: {val:.1f} pts)"),
    ("rolling_last5_reb",      "Strong recent rebounding (last 5 avg: {val:.1f})",
                               "Weak rebounding recently (last 5 avg: {val:.1f})"),
    ("rolling_last10_reb",     "High rebounding baseline (last 10 avg: {val:.1f})",
                               "Low rebounding baseline (last 10 avg: {val:.1f})"),
    ("rolling_last5_ast",      "Strong recent assists (last 5 avg: {val:.1f})",
                               "Fewer assists recently (last 5 avg: {val:.1f})"),
    ("rolling_last10_ast",     "High assist baseline (last 10 avg: {val:.1f})",
                               "Low assist baseline (last 10 avg: {val:.1f})"),
    ("rolling_last5_minutes",  "Playing heavy minutes recently ({val:.0f} avg)",
                               "Lighter minutes load recently ({val:.0f} avg)"),
    ("rolling_last5_minutes_std", "Consistent minutes (low variance: {val:.1f})",
                                  "Inconsistent minutes (high variance: {val:.1f})"),
    # Season averages
    ("season_avg_pts",         "Strong season scoring average ({val:.1f} ppg)",
                               "Modest season scoring average ({val:.1f} ppg)"),
    ("season_avg_reb",         "Strong season rebounding ({val:.1f} rpg)",
                               "Modest season rebounding ({val:.1f} rpg)"),
    ("season_avg_ast",         "High season assist rate ({val:.1f} apg)",
                               "Modest season assist rate ({val:.1f} apg)"),
    ("season_avg_minutes",     "Averaging heavy minutes this season ({val:.0f})",
                               "Averaging lighter minutes this season ({val:.0f})"),
    ("season_avg_usage_proxy", "High usage rate this season",
                               "Lower usage rate this season"),
    # Opponent context
    ("opp_def_rating",         "Facing weak defense (rated {val:.1f})",
                               "Facing elite defense (rated {val:.1f})"),
    ("opp_def_rank",           "Opponent is a weak defensive team (rank {val:.0f}/30)",
                               "Opponent is a strong defensive team (rank {val:.0f}/30)"),
    # Game context
    # back_to_back: label based on feature VALUE (0=rested, 1=B2B), not SHAP direction
    # We handle this as a special case in _match_label() below
    ("back_to_back",           "Well rested (not a back-to-back)",
                               "Back-to-back game (fatigue factor)"),
    ("home_game",              "Home court advantage",
                               "Away game"),
    ("rest_days_capped",       "Extended rest ({val:.0f} days)",
                               "Limited rest ({val:.0f} days)"),
    ("team_pace",              "Fast-paced team (more possessions: {val:.1f})",
                               "Slow-paced team (fewer possessions: {val:.1f})"),
    # Trend
    ("trend_pts",              "Scoring is trending up recently",
                               "Scoring is trending down recently"),
    ("trend_fantasy_score",    "Overall performance trending up",
                               "Overall performance trending down"),
    # Opponent history
    ("vs_opp_rolling3_pts",    "Historically scores well vs this opponent",
                               "Historically struggles vs this opponent"),
    ("vs_opp_rolling3_reb",    "Historically rebounds well vs this opponent",
                               "Historically fewer rebounds vs this opponent"),
]


def _match_label(feature_name: str, shap_val: float,
                 feature_val: float) -> str | None:
    """
    Find the human-readable label for a feature based on its name and SHAP direction.

    For binary features (back_to_back, home_game) we use the feature VALUE
    to determine the label, not the SHAP direction — a back-to-back is always
    a back-to-back regardless of which direction it pushed the prediction.

    Returns None if no matching label is found (unnamed features are skipped).
    """
    BINARY_FEATURES = {"back_to_back", "home_game"}

    for pattern, pos_label, neg_label in FEATURE_LABELS:
        if pattern in feature_name:
            # Binary features: use feature value to pick label
            if feature_name in BINARY_FEATURES:
                template = neg_label if feature_val > 0.5 else pos_label
            else:
                template = pos_label if shap_val > 0 else neg_label
            try:
                return template.format(val=feature_val)
            except (KeyError, ValueError):
                return template
    return None


def generate_reasoning(
    shap_values: np.ndarray,
    feature_names: list,
    feature_values: np.ndarray,
    target: str,
    n_reasons: int = 5,
) -> list[dict]:
    """
    Convert SHAP values into a ranked list of reasons for a prediction.

    Args:
        shap_values   : 1D array of SHAP values (one per feature)
        feature_names : list of feature names matching shap_values
        feature_values: 1D array of actual feature values for this row
        target        : stat being predicted (e.g. "pts")
        n_reasons     : how many reasons to return

    Returns:
        List of dicts: [{"direction": "+"/"-", "label": str, "shap": float}, ...]
        Sorted by absolute SHAP value descending (most impactful first).
    """
    # Sort features by absolute SHAP value
    pairs = sorted(
        zip(feature_names, shap_values, feature_values),
        key=lambda x: abs(x[1]),
        reverse=True,
    )

    reasons = []
    for feat, shap_val, feat_val in pairs:
        if abs(shap_val) < 0.05:
            break  # Stop at negligible contributions

        label = _match_label(feat, shap_val, float(feat_val))
        if label is None:
            continue  # Skip features with no human label

        reasons.append({
            "direction" : "+" if shap_val > 0 else "-",
            "label"     : label,
            "shap_value": round(float(shap_val), 3),
            "feature"   : feat,
        })

        if len(reasons) >= n_reasons:
            break

    return reasons


def format_reasoning_text(all_reasons: dict[str, list]) -> str:
    """
    Format reasoning from all models into the final output string.

    Deduplicates labels that appear across multiple models and
    formats them as the +/- bullet list shown in the spec.

    Args:
        all_reasons: {"pts": [...reasons...], "reb": [...], ...}

    Returns:
        Multi-line string ready to display to the user.
    """
    # Collect all reasons, deduplicate by label
    seen_labels = set()
    positive_reasons = []
    negative_reasons = []

    # Weight pts/reb/ast more heavily in the combined reasoning
    # since they dominate the fantasy score
    priority_targets = ["pts", "reb", "ast", "minutes", "stl", "blk"]

    for target in priority_targets:
        if target not in all_reasons:
            continue
        for r in all_reasons[target]:
            if r["label"] in seen_labels:
                continue
            seen_labels.add(r["label"])
            if r["direction"] == "+":
                positive_reasons.append(r["label"])
            else:
                negative_reasons.append(r["label"])

    lines = ["Reasoning:"]
    lines.append("")

    for label in positive_reasons[:4]:
        lines.append(f"+ {label}")
    for label in negative_reasons[:3]:
        lines.append(f"- {label}")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Fantasy score
# ─────────────────────────────────────────────────────────────────────────────

def compute_fantasy_score(predictions: dict, tov: float = 0.0) -> float:
    """Compute DraftKings fantasy score from predicted stats."""
    return (
        predictions.get("pts", 0)   * 1.0
        + predictions.get("reb", 0) * 1.2
        + predictions.get("ast", 0) * 1.5
        + predictions.get("stl", 0) * 3.0
        + predictions.get("blk", 0) * 3.0
        - tov                       * 1.0
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main prediction + explanation function
# ─────────────────────────────────────────────────────────────────────────────

def explain_prediction(
    feature_row: pd.DataFrame,
    models: dict,
    feature_names: list,
    n_reasons: int = 5,
) -> dict:
    """
    Generate predictions + SHAP explanations for a single player/game row.

    Args:
        feature_row  : DataFrame with exactly one row, columns = feature_names
        models       : dict of {target: XGBRegressor}
        feature_names: list of feature column names
        n_reasons    : reasons to show per stat (deduped across stats)

    Returns:
        {
            "predictions"   : {"pts": 27.1, "reb": 8.3, ...},
            "fantasy_score" : 44.6,
            "all_reasons"   : {"pts": [...], "reb": [...], ...},
            "reasoning_text": "Reasoning:\n+ Strong recent scoring..."
        }
    """
    X = feature_row[feature_names].copy()

    # Fill any NaN features with 0 — XGBoost handles NaN internally
    # but explicit fill avoids SHAP warnings
    X = X.fillna(X.median() if len(X) > 1 else 0)

    predictions = {}
    all_reasons = {}

    for target, model in models.items():
        # Predict
        pred = float(model.predict(X)[0])
        pred = max(0.0, round(pred, 1))   # clip negatives (can't score -2 pts)
        predictions[target] = pred

        # SHAP
        shap_vals = compute_shap_values(model, X)
        feat_vals = X.values[0]

        reasons = generate_reasoning(
            shap_vals, feature_names, feat_vals, target, n_reasons
        )
        all_reasons[target] = reasons

    # Fantasy score
    fantasy = compute_fantasy_score(predictions)

    # Combined reasoning text
    reasoning_text = format_reasoning_text(all_reasons)

    return {
        "predictions"   : predictions,
        "fantasy_score" : round(fantasy, 1),
        "all_reasons"   : all_reasons,
        "reasoning_text": reasoning_text,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Pretty print
# ─────────────────────────────────────────────────────────────────────────────

def print_prediction(result: dict, player_name: str = "Player") -> None:
    """Print a formatted prediction card."""
    preds = result["predictions"]
    print(f"\n{'═'*45}")
    print(f" Prediction: {player_name}")
    print(f"{'═'*45}")
    print(f"\n Predicted Stats:")
    print(f"   Points  : {preds.get('pts', '?')}")
    print(f"   Rebounds: {preds.get('reb', '?')}")
    print(f"   Assists : {preds.get('ast', '?')}")
    print(f"   Steals  : {preds.get('stl', '?')}")
    print(f"   Blocks  : {preds.get('blk', '?')}")
    print(f"   Minutes : {preds.get('minutes', '?')}")
    print(f"\n Fantasy Score: {result['fantasy_score']}")
    print(f"\n{result['reasoning_text']}")
    print(f"{'═'*45}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Standalone test — runs on a real row from the features dataset
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    log.info("Loading models and features for test prediction...")
    models = load_models()
    feature_names = load_feature_names()

    # Load features dataset and add minutes std if missing
    df = pd.read_csv(PROCESSED / "features.csv", parse_dates=["game_date"])
    if "rolling_last5_minutes_std" not in df.columns:
        df["rolling_last5_minutes_std"] = (
            df.groupby("player_id")["minutes"]
            .transform(lambda x: x.shift(1).rolling(5, min_periods=2).std())
            .fillna(3.5)
        )

    # Filter feature_names to only those present in df
    feature_names = [f for f in feature_names if f in df.columns]

    # Pick a test row — find a LeBron game with full features
    test_candidates = df[
        (df["player_name"] == "LeBron James") &
        df[feature_names].notna().all(axis=1)
    ]

    if test_candidates.empty:
        log.warning("No complete LeBron rows found — using first complete row")
        test_candidates = df[df[feature_names].notna().all(axis=1)]

    test_row = test_candidates.iloc[-1]   # most recent game
    player   = test_row.get("player_name", "Unknown")
    date     = str(test_row.get("game_date", ""))[:10]
    actual   = {t: test_row.get(t) for t in TARGETS}

    log.info(f"Test row: {player} on {date}")
    log.info(f"Actual stats: {actual}")

    result = explain_prediction(
        test_row.to_frame().T,
        models,
        feature_names,
    )

    print_prediction(result, f"{player} ({date})")

    print("Actual stats for comparison:")
    for stat in TARGETS:
        actual_val = actual.get(stat, "?")
        pred_val   = result["predictions"].get(stat, "?")
        try:
            diff = abs(float(actual_val) - float(pred_val))
            print(f"  {stat:10s}: predicted {pred_val:5.1f}  actual {actual_val:5.1f}"
                  f"  diff {diff:.1f}")
        except (TypeError, ValueError):
            print(f"  {stat:10s}: predicted {pred_val}  actual {actual_val}")