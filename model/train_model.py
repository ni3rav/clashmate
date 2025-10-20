import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from imblearn.over_sampling import SMOTE
import pickle
import re
from tabulate import tabulate


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


def remove_duplicates(df):
    """Remove duplicate card entries, keeping the first occurrence"""
    print("\n=== DATA CLEANING: Remove Duplicates ===")
    print(f"Before: {len(df)} rows")
    df_dedup = df.drop_duplicates(subset=["name"], keep="first")
    duplicates_removed = len(df) - len(df_dedup)
    print(f"After: {len(df_dedup)} rows")
    print(f"Removed: {duplicates_removed} duplicate cards")
    return df_dedup


def filter_valid_cards(df):
    """Filter cards with valid elixir costs and basic stats"""
    print("\n=== DATA CLEANING: Filter Valid Cards ===")
    print(f"Before: {len(df)} rows")

    # Only keep cards with valid elixir
    df_valid = df[df["elixir_clean"].notna()].copy()

    # Remove spawner-only cards (those with type containing "Spawner")
    df_valid = df_valid[~df_valid["type"].str.contains("Spawner", na=False)]

    print(f"After: {len(df_valid)} rows")
    print(f"Removed: {len(df) - len(df_valid)} invalid/spawner cards")
    return df_valid


def show_data_summary(df, stage=""):
    """Display summary statistics of the dataset"""
    print(f"\n=== DATA SUMMARY {stage} ===")
    print(f"Total rows: {len(df)}")
    print(f"Card types: {df['type'].unique().tolist()}")
    print(f"Type distribution:\n{df['type'].value_counts()}")
    if "elixir_clean" in df.columns:
        print(
            f"Elixir range: {df['elixir_clean'].min():.0f} - {df['elixir_clean'].max():.0f}"
        )
        print(f"Elixir distribution:\n{df['elixir_clean'].value_counts().sort_index()}")


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


def apply_smote(X, y):
    """Apply SMOTE to augment training data"""
    print("\n=== DATA AUGMENTATION: SMOTE ===")
    print(f"Before SMOTE: {len(X)} samples")
    print(f"Elixir distribution before:\n{pd.Series(y).value_counts().sort_index()}")

    # SMOTE for regression: bin the target for stratification
    y_binned = pd.cut(y, bins=5, labels=False)

    try:
        smote = SMOTE(random_state=42, k_neighbors=3)
        X_resampled, y_binned_resampled = smote.fit_resample(X, y_binned)

        # Map back to original y values by finding closest matches
        y_resampled = []
        for idx in range(len(X_resampled)):
            bin_val = y_binned_resampled[idx]
            # Get original y values from same bin
            original_in_bin = y[y_binned == bin_val]
            if len(original_in_bin) > 0:
                y_resampled.append(np.random.choice(original_in_bin))
            else:
                y_resampled.append(y.median())

        y_resampled = pd.Series(y_resampled, index=range(len(y_resampled)))

        print(f"After SMOTE: {len(X_resampled)} samples")
        print(f"Elixir distribution after:\n{y_resampled.value_counts().sort_index()}")
        print(f"Synthetic samples created: {len(X_resampled) - len(X)}")

        return X_resampled, y_resampled
    except Exception as e:
        print(f"SMOTE failed: {e}")
        print("Continuing without SMOTE...")
        return X, y


def train_and_evaluate(X, y, use_smote=True):
    print(f"\n=== MODEL TRAINING ===")
    print(f"Initial dataset: {len(X)} samples")
    print(f"Features: {list(X.columns)}")
    print(f"Target range: {y.min():.0f} - {y.max():.0f} elixir")

    # Split before SMOTE to avoid data leakage
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"\nTrain set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")

    # Apply SMOTE only to training data
    if use_smote and len(X_train) >= 10:
        X_train, y_train = apply_smote(X_train, y_train)

    # Hyperparameter tuning
    print("\nRunning GridSearchCV...")
    param_grid = {"n_estimators": [50, 100, 200], "max_depth": [5, 10, 20]}
    cv = KFold(n_splits=min(3, len(X_train) // 10), shuffle=True, random_state=42)

    grid = GridSearchCV(
        RandomForestRegressor(random_state=42),
        param_grid,
        cv=cv,
        scoring="r2",
        n_jobs=-1,
    )
    grid.fit(X_train, y_train)

    print(f"Best parameters: {grid.best_params_}")
    print(f"Best CV R^2: {grid.best_score_:.3f}")

    # Train final model
    print("\nTraining final model...")
    model = RandomForestRegressor(**grid.best_params_, random_state=42)
    model.fit(X_train, y_train)

    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Calculate comprehensive metrics
    metrics = calculate_metrics(y_train, y_train_pred, y_test, y_test_pred)
    display_metrics_table(metrics)
    display_feature_importance(model, X.columns)

    return model, grid, X_train, X_test, y_train, y_test, y_test_pred


def calculate_metrics(y_train, y_train_pred, y_test, y_test_pred):
    """Calculate comprehensive evaluation metrics"""
    return {
        "Train R^2": r2_score(y_train, y_train_pred),
        "Test R^2": r2_score(y_test, y_test_pred),
        "Train RMSE": np.sqrt(mean_squared_error(y_train, y_train_pred)),
        "Test RMSE": np.sqrt(mean_squared_error(y_test, y_test_pred)),
        "Train MAE": mean_absolute_error(y_train, y_train_pred),
        "Test MAE": mean_absolute_error(y_test, y_test_pred),
        "Train MAPE": np.mean(np.abs((y_train - y_train_pred) / y_train)) * 100,
        "Test MAPE": np.mean(np.abs((y_test - y_test_pred) / y_test)) * 100,
    }


def display_metrics_table(metrics):
    """Display metrics in a nice table format"""
    print("\n" + "=" * 60)
    print("MODEL EVALUATION METRICS")
    print("=" * 60)

    table_data = []
    for metric, value in metrics.items():
        dataset = "Training" if "Train" in metric else "Testing"
        metric_name = metric.replace("Train ", "").replace("Test ", "")
        table_data.append([dataset, metric_name, f"{value:.4f}"])

    headers = ["Dataset", "Metric", "Value"]
    print(tabulate(table_data, headers=headers, tablefmt="grid"))
    print("=" * 60)


def display_feature_importance(model, feature_names):
    """Display feature importance in table format"""
    importance_df = pd.DataFrame(
        {"Feature": feature_names, "Importance": model.feature_importances_}
    ).sort_values("Importance", ascending=False)

    print("\n" + "=" * 50)
    print("FEATURE IMPORTANCE")
    print("=" * 50)
    print(
        tabulate(
            importance_df,
            headers=["Feature", "Importance"],
            tablefmt="grid",
            showindex=False,
            floatfmt=".4f",
        )
    )
    print("=" * 50)


def save_artifacts(model, le, X, feature_cols):
    with open("elixir_model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("type_encoder.pkl", "wb") as f:
        pickle.dump(le, f)
    medians = {col: X[col].median() for col in feature_cols}
    with open("feature_medians.pkl", "wb") as f:
        pickle.dump(medians, f)
    print("Model, encoder, and medians saved.")


def show_sample_predictions(model, X_test, y_test, y_test_pred):
    """Display sample predictions in table format"""
    print("\n" + "=" * 60)
    print("SAMPLE PREDICTIONS")
    print("=" * 60)

    sample_data = []
    for i in range(min(10, len(X_test))):
        actual = y_test.iloc[i]
        pred = y_test_pred[i]
        error = abs(actual - pred)
        error_pct = (error / actual) * 100
        sample_data.append(
            [i + 1, f"{actual:.1f}", f"{pred:.1f}", f"{error:.2f}", f"{error_pct:.1f}%"]
        )

    headers = ["#", "Actual", "Predicted", "Error", "Error %"]
    print(tabulate(sample_data, headers=headers, tablefmt="grid"))
    print("=" * 60)


def main():
    print("=" * 60)
    print("CLASHMATE - ELIXIR COST PREDICTOR")
    print("Model Training Pipeline")
    print("=" * 60)

    # Load data
    print("\n=== LOADING DATA ===")
    df = pd.read_csv("cards.csv")
    print(f"Loaded: {len(df)} rows")
    print(f"Columns: {', '.join(df.columns.tolist())}")

    # Extract features
    df = add_features(df)
    show_data_summary(df, "(RAW)")

    # Data cleaning
    df = remove_duplicates(df)
    df = filter_valid_cards(df)
    show_data_summary(df, "(CLEANED)")

    # Prepare training data
    print("\n=== PREPARING FEATURES ===")
    X, y, le, feature_cols = prepare_training_data(df)
    print(f"Final dataset: {len(X)} samples, {len(feature_cols)} features")

    # Train and evaluate
    model, grid, X_train, X_test, y_train, y_test, y_test_pred = train_and_evaluate(
        X, y, use_smote=True
    )

    # Save artifacts
    print("\n=== SAVING MODEL ===")
    save_artifacts(model, le, X, feature_cols)

    # Show predictions
    show_sample_predictions(model, X_test, y_test, y_test_pred)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
