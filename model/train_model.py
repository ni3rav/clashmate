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


def compute_cycle_card_proxy(df):
    """
    Compute a non-leaky proxy for cycle cards based on input features only.
    Cycle cards typically have low HP, low damage, low DPS, small count, and fast hit speed.
    This feature can be computed at inference time without knowing the elixir cost.
    """
    # Fill missing values with median for threshold calculations
    hp_median = df["hitpoints_clean"].median()
    damage_median = df["damage_clean"].median()
    dps_median = df["dps_clean"].median()
    
    # Cycle cards are characterized by:
    # 1. Low HP (below median)
    # 2. Low damage (below median) 
    # 3. Low DPS (below median)
    # 4. Small count (≤ 2, or missing/1)
    # 5. Fast hit speed (low hitSpeed, but this is less reliable)
    
    count_filled = df["count_clean"].fillna(1)
    
    # Create proxy: card is likely a cycle card if it has low stats across multiple dimensions
    is_cycle_proxy = (
        (df["hitpoints_clean"].fillna(hp_median) < hp_median) &
        (df["damage_clean"].fillna(damage_median) < damage_median) &
        (df["dps_clean"].fillna(dps_median) < dps_median) &
        (count_filled <= 2)
    ).astype(int)
    
    return is_cycle_proxy


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

    # Add cycle card proxy (NON-LEAKY: computed from input features only)
    # This replaces the previous target-derived feature that caused data leakage
    df_clean["is_cycle_card"] = compute_cycle_card_proxy(df_clean)

    # Add HP/Damage ratio feature
    df_clean["hp_damage_ratio"] = df_clean["hitpoints_clean"] / (
        df_clean["damage_clean"] + 1
    )

    feature_cols = [
        "type_encoded",
        "hitpoints_clean",
        "damage_clean",
        "hitSpeed_clean",
        "dps_clean",
        "range_clean",
        "count_clean",
        "has_area_damage",
        "is_cycle_card",
        "hp_damage_ratio",
        # Removed: has_spawned_unit (0% importance)
    ]
    X = df_clean[feature_cols].copy()
    for col in X.columns:
        X[col] = X[col].fillna(X[col].median())
    y = df_clean["elixir_clean"]
    return X, y, le, feature_cols


def apply_smote(X, y):
    """Apply SMOTE to augment training data"""
    print("\n=== DATA AUGMENTATION: SMOTE ===")
    print(f"Before SMOTE: {len(X)} samples")
    print(f"Elixir distribution before:\n{pd.Series(y).value_counts().sort_index()}")

    # Check minimum samples per class
    y_counts = pd.Series(y).value_counts()
    min_samples = y_counts.min()

    # Calculate safe k_neighbors value
    # Need at least k_neighbors + 1 samples in smallest class
    safe_k = max(1, min(3, min_samples - 1))

    if safe_k < 1:
        print(f"SMOTE skipped: Smallest class has only {min_samples} sample(s)")
        print("Need at least 2 samples per class for SMOTE")
        return X, y

    # SMOTE for regression: bin the target for stratification
    y_binned = pd.cut(y, bins=3, labels=False)  # Reduced bins from 5 to 3

    try:
        smote = SMOTE(random_state=42, k_neighbors=safe_k)
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

        print(f"After SMOTE: {len(X_resampled)} samples (k_neighbors={safe_k})")
        print(f"Elixir distribution after:\n{y_resampled.value_counts().sort_index()}")
        print(f"Synthetic samples created: {len(X_resampled) - len(X)}")

        return X_resampled, y_resampled
    except Exception as e:
        print(f"SMOTE failed: {e}")
        print("Continuing without SMOTE...")
        return X, y


def augment_with_jitter(
    X, y, n_augment=1, exclude_cols=None, scale=0.05, random_state=42
):
    """Augment dataset by adding small Gaussian noise to numeric features.

    - n_augment: how many augmented copies to create (per original dataset)
    - exclude_cols: list of columns NOT to jitter (categorical/binary)
    - scale: fraction of column median used as std for noise (e.g. 0.05 = 5%)
    """
    print("\n=== DATA AUGMENTATION: JITTER ===")
    if exclude_cols is None:
        exclude_cols = ["type_encoded", "is_cycle_card"]

    rng = np.random.RandomState(random_state)
    X_list = [X.copy()]
    y_list = [y.copy()]

    # Determine numeric columns to jitter (exclude categorical/binary)
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    jitter_cols = [c for c in numeric_cols if c not in exclude_cols]

    if len(jitter_cols) == 0:
        print("No numeric columns found to jitter. Skipping augmentation.")
        return X, y

    for i in range(n_augment):
        Xj = X.copy()
        for col in jitter_cols:
            # Use median as scale reference, fallback to std or 1.0
            med = Xj[col].median()
            if med == 0 or pd.isna(med):
                med = (
                    Xj[col].std()
                    if not pd.isna(Xj[col].std()) and Xj[col].std() > 0
                    else 1.0
                )
            std = abs(scale * med)
            noise = rng.normal(loc=0.0, scale=std, size=len(Xj))
            Xj[col] = Xj[col] + noise

            # Clamp sensible columns to non-negative
            if col in [
                "hitpoints_clean",
                "damage_clean",
                "dps_clean",
                "range_clean",
                "count_clean",
                "hp_damage_ratio",
                "hitSpeed_clean",
            ]:
                Xj[col] = Xj[col].clip(lower=0)

        X_list.append(Xj)
        y_list.append(y.copy())

    X_aug = pd.concat(X_list, ignore_index=True)
    y_aug = pd.concat(y_list, ignore_index=True)

    print(f"Original samples: {len(X)}")
    print(
        f"Augmented samples: {len(X_aug)} (added {len(X_aug)-len(X)} rows via jitter)"
    )
    return X_aug, y_aug


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

    # Apply SMOTE only to training data (optional - can introduce noise)
    if use_smote and len(X_train) >= 10:
        X_train, y_train = apply_smote(X_train, y_train)
    else:
        print("\n=== SMOTE DISABLED ===")
        print("Training with original data only")

    # Hyperparameter tuning with balanced regularization
    print("\nRunning GridSearchCV...")
    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [5, 7, 10],  # Slightly relaxed from [3, 5, 10]
        "min_samples_split": [5, 10],  # Reduced from [5, 10, 20]
        "min_samples_leaf": [2, 4],  # Reduced from [2, 4, 8]
    }
    cv = KFold(
        n_splits=min(5, max(3, len(X_train) // 20)), shuffle=True, random_state=42
    )

    grid = GridSearchCV(
        RandomForestRegressor(random_state=42),
        param_grid,
        cv=cv,
        scoring="r2",
        n_jobs=-1,
        verbose=1,
    )
    grid.fit(X_train, y_train)

    print(f"Best parameters: {grid.best_params_}")
    print(f"Best CV R^2: {grid.best_score_:.3f}")

    # Train final model with regularization
    print("\nTraining final model with regularization...")
    model = RandomForestRegressor(
        **grid.best_params_,
        random_state=42,
        max_features="sqrt",  # Use sqrt of features to reduce overfitting
    )
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


def save_artifacts(model, le, X, feature_cols, proxy_medians=None):
    with open("elixir_model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("type_encoder.pkl", "wb") as f:
        pickle.dump(le, f)
    medians = {col: X[col].median() for col in feature_cols}
    # Also save medians used for cycle card proxy computation
    if proxy_medians:
        medians.update(proxy_medians)
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


def show_worst_predictions(X_test, y_test, y_test_pred, df_clean, n=10):
    """Display worst predictions for error analysis"""
    print("\n" + "=" * 60)
    print(f"TOP {n} WORST PREDICTIONS (Error Analysis)")
    print("=" * 60)

    # Calculate errors
    errors = np.abs(y_test.values - y_test_pred)
    error_indices = np.argsort(errors)[-n:][::-1]  # Top N worst

    worst_data = []
    for idx in error_indices:
        test_idx = y_test.index[idx]
        actual = y_test.iloc[idx]
        pred = y_test_pred[idx]
        error = errors[idx]
        error_pct = (error / actual) * 100

        # Get card info from test data (avoid looking up augmented indices)
        # Use original index modulo to map back to original cards
        original_idx = test_idx % len(df_clean)
        card_name = (
            df_clean.iloc[original_idx]["name"]
            if original_idx < len(df_clean)
            else "Augmented"
        )
        card_type = (
            df_clean.iloc[original_idx]["type"]
            if original_idx < len(df_clean)
            else "Unknown"
        )

        worst_data.append(
            [
                card_name[:20],  # Truncate long names
                card_type,
                f"{actual:.1f}",
                f"{pred:.1f}",
                f"{error:.2f}",
                f"{error_pct:.1f}%",
            ]
        )

    headers = ["Card", "Type", "Actual", "Predicted", "Error", "Error %"]
    print(tabulate(worst_data, headers=headers, tablefmt="grid"))
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
    print(f"Features: {feature_cols}")

    # Compute and save medians for cycle card proxy (needed for inference)
    # These medians are computed from the original (non-augmented) data
    proxy_medians = {
        "proxy_hp_median": X["hitpoints_clean"].median(),
        "proxy_damage_median": X["damage_clean"].median(),
        "proxy_dps_median": X["dps_clean"].median(),
    }

    # Apply jitter augmentation to increase dataset size
    # For academic purposes: n_augment=4 creates 4 additional copies with 5% Gaussian noise
    # This is a standard technique for small datasets to improve generalization
    # Results in 5x original data (~550 samples from ~110 base)
    X, y = augment_with_jitter(X, y, n_augment=4, scale=0.05)

    # Train and evaluate (SMOTE disabled - it was adding noise)
    model, grid, X_train, X_test, y_train, y_test, y_test_pred = train_and_evaluate(
        X, y, use_smote=False
    )

    # Save artifacts
    print("\n=== SAVING MODEL ===")
    save_artifacts(model, le, X, feature_cols, proxy_medians=proxy_medians)

    # Show predictions
    show_sample_predictions(model, X_test, y_test, y_test_pred)

    # Show worst predictions for error analysis
    show_worst_predictions(X_test, y_test, y_test_pred, df, n=10)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
