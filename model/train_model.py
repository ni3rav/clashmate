import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import pickle
import re


def extract_elixir(val):
    if pd.isna(val) or val == "N/A":
        return None
    match = re.search(r"(\d+)", str(val))
    return float(match.group(1)) if match else None


def clean_numeric(val):
    if pd.isna(val) or val == "N/A" or val == "":
        return None
    val_str = str(val).replace(",", "")
    numbers = re.findall(r"\d+\.?\d*", val_str)
    if not numbers:
        return None
    floats = [float(n) for n in numbers]
    return np.mean(floats) if len(floats) > 1 else floats[0]


def add_features(df):
    df["elixir_clean"] = df["elixir"].apply(extract_elixir)
    numeric_cols = ["hitpoints", "damage", "hitSpeed", "dps", "range", "count"]
    for col in numeric_cols:
        if col in df.columns:
            df[f"{col}_clean"] = df[col].apply(clean_numeric)
    return df


def add_binary_features(df):
    df["has_area_damage"] = df["areaDamage"].apply(
        lambda x: 0 if pd.isna(x) or x in ["N/A", "", None] else 1
    )
    df["has_spawned_unit"] = df["spawnedUnit"].apply(
        lambda x: 0 if pd.isna(x) or x in ["N/A", "", None] else 1
    )
    return df


def prepare_training_data(df):
    df_clean = df[df["elixir_clean"].notna()].copy()
    le = LabelEncoder()
    df_clean["type_encoded"] = le.fit_transform(df_clean["type"])
    df_clean = add_binary_features(df_clean)
    feature_cols = [
        "type_encoded",
        "hitpoints_clean",
        "damage_clean",
        "hitSpeed_clean",
        "dps_clean",
        "range_clean",
        "count_clean",
        "has_area_damage",
        "has_spawned_unit",
    ]
    X = df_clean[feature_cols].copy()
    for col in X.columns:
        X[col].fillna(X[col].median(), inplace=True)
    y = df_clean["elixir_clean"]
    return X, y, le, feature_cols


def train_and_evaluate(X, y):
    print(f"Loaded {len(X)} cards for training")
    print(f"Training features: {list(X.columns)}")
    print(f"Target: elixir cost (range: {y.min():.0f}-{y.max():.0f})")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print("Running GridSearchCV for Random Forest hyperparameters...")
    param_grid = {"n_estimators": [50, 100, 200], "max_depth": [5, 10, 20]}
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    grid = GridSearchCV(
        RandomForestRegressor(random_state=42),
        param_grid,
        cv=cv,
        scoring="r2",
        n_jobs=-1,
    )
    grid.fit(X_train, y_train)
    print(f"Best parameters: {grid.best_params_}")
    print(f"Best cross-validated R^2: {grid.best_score_:.3f}")
    print("Training final Random Forest model...")
    model = RandomForestRegressor(**grid.best_params_, random_state=42)
    model.fit(X_train, y_train)
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    print(f"Model trained!")
    print(f"   Train R^2 score: {train_score:.3f}")
    print(f"   Test R^2 score: {test_score:.3f}")
    importance = pd.DataFrame(
        {"feature": X.columns, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)
    print("Feature Importance:")
    for _, row in importance.iterrows():
        print(f"   {row['feature']}: {row['importance']:.3f}")
    return model, grid, X_train, X_test, y_train, y_test


def save_artifacts(model, le, X, feature_cols):
    with open("elixir_model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("type_encoder.pkl", "wb") as f:
        pickle.dump(le, f)
    medians = {col: X[col].median() for col in feature_cols}
    with open("feature_medians.pkl", "wb") as f:
        pickle.dump(medians, f)
    print("Model, encoder, and medians saved.")


def show_sample_predictions(model, X_test, y_test):
    print("Sample predictions:")
    for i in range(min(5, len(X_test))):
        actual = y_test.iloc[i]
        pred = model.predict([X_test.iloc[i]])[0]
        print(
            f"   Actual: {actual:.1f} | Predicted: {pred:.1f} | Diff: {abs(actual-pred):.2f}"
        )


def main():
    df = pd.read_csv("cards.csv")
    print(f"Columns: {df.columns.tolist()}")
    df = add_features(df)
    X, y, le, feature_cols = prepare_training_data(df)
    model, grid, X_train, X_test, y_train, y_test = train_and_evaluate(X, y)
    save_artifacts(model, le, X, feature_cols)
    show_sample_predictions(model, X_test, y_test)


if __name__ == "__main__":
    main()
