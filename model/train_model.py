import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.preprocessing import LabelEncoder
import pickle
import re

# Load the CSV
df = pd.read_csv("cards.csv")

print(f"📊 Loaded {len(df)} cards")
print(f"Columns: {df.columns.tolist()}")


# Clean elixir column - extract just the number
def extract_elixir(val):
    if pd.isna(val) or val == "N/A":
        return None
    # Extract first number from strings like "5 (1)" or "6/3"
    match = re.search(r"(\d+)", str(val))
    return float(match.group(1)) if match else None


df["elixir_clean"] = df["elixir"].apply(extract_elixir)


# Clean numeric columns - remove commas and extract numbers
def clean_numeric(val):
    if pd.isna(val) or val == "N/A" or val == "":
        return None
    # Remove commas
    val_str = str(val).replace(",", "")
    # Extract all numbers (handles '202/133', '84 (x10)', '1440 (1,200+240)', '35-422', etc.)
    numbers = re.findall(r"\d+\.?\d*", val_str)
    if not numbers:
        return None
    # Convert all to float
    floats = [float(n) for n in numbers]
    # Use mean if multiple values, else single value
    return np.mean(floats) if len(floats) > 1 else floats[0]


numeric_cols = ["hitpoints", "damage", "hitSpeed", "dps", "range", "count"]
for col in numeric_cols:
    if col in df.columns:
        df[f"{col}_clean"] = df[col].apply(clean_numeric)

# Drop rows with no elixir cost
df_clean = df[df["elixir_clean"].notna()].copy()

print(f"\n✅ Cleaned data: {len(df_clean)} cards with valid elixir cost")

# Encode card type
le = LabelEncoder()
df_clean["type_encoded"] = le.fit_transform(df_clean["type"])

# Add binary features for area damage and spawned units
df_clean["has_area_damage"] = df_clean["areaDamage"].apply(
    lambda x: 0 if pd.isna(x) or x in ["N/A", "", None] else 1
)
df_clean["has_spawned_unit"] = df_clean["spawnedUnit"].apply(
    lambda x: 0 if pd.isna(x) or x in ["N/A", "", None] else 1
)

# Features for training
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

# Fill NaN with median for each column
X = df_clean[feature_cols].copy()
for col in X.columns:
    X[col].fillna(X[col].median(), inplace=True)

y = df_clean["elixir_clean"]

print(f"\n🎯 Training features: {feature_cols}")
print(f"Target: elixir cost (range: {y.min():.0f}-{y.max():.0f})")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Cross-validation and hyperparameter tuning
print("\n🔍 Running GridSearchCV for Random Forest hyperparameters...")
param_grid = {"n_estimators": [50, 100, 200], "max_depth": [5, 10, 20]}
cv = KFold(n_splits=5, shuffle=True, random_state=42)
grid = GridSearchCV(
    RandomForestRegressor(random_state=42), param_grid, cv=cv, scoring="r2", n_jobs=-1
)
grid.fit(X_train, y_train)

print(f"Best parameters: {grid.best_params_}")
print(f"Best cross-validated R²: {grid.best_score_:.3f}")

# Train final model with best params
print("\n🤖 Training final Random Forest model...")
model = RandomForestRegressor(**grid.best_params_, random_state=42)
model.fit(X_train, y_train)

# Evaluate
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)

print(f"✅ Model trained!")
print(f"   Train R² score: {train_score:.3f}")
print(f"   Test R² score: {test_score:.3f}")

# Feature importance
importance = pd.DataFrame(
    {"feature": feature_cols, "importance": model.feature_importances_}
).sort_values("importance", ascending=False)

print(f"\n📊 Feature Importance:")
for _, row in importance.iterrows():
    print(f"   {row['feature']}: {row['importance']:.3f}")

# Save model and encoder
with open("elixir_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("type_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

# Save feature medians for filling NaN during prediction
medians = {col: X[col].median() for col in feature_cols}
with open("feature_medians.pkl", "wb") as f:
    pickle.dump(medians, f)

print("\n💾 Model saved as 'elixir_model.pkl'")
print("💾 Encoder saved as 'type_encoder.pkl'")
print("💾 Medians saved as 'feature_medians.pkl'")

# Test predictions on a few examples
print("\n🧪 Sample predictions:")
for i in range(min(5, len(X_test))):
    actual = y_test.iloc[i]
    pred = model.predict([X_test.iloc[i]])[0]
    print(
        f"   Actual: {actual:.1f} | Predicted: {pred:.1f} | Diff: {abs(actual-pred):.2f}"
    )
