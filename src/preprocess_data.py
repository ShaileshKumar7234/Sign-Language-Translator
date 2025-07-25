import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

RAW_DIR = 'data/raw'
PROCESSED_DIR = 'data/processed'
os.makedirs(PROCESSED_DIR, exist_ok=True)

# Combine all CSVs
all_data = []
for file in os.listdir(RAW_DIR):
    if file.endswith('.csv'):
        print(f"[INFO] Loading {file}")
        df = pd.read_csv(os.path.join(RAW_DIR, file))
        all_data.append(df)

# Concatenate and clean
df = pd.concat(all_data, ignore_index=True)
print(f"[INFO] Total samples: {len(df)}")

# Separate features and labels
X = df.iloc[:, :-1]  # landmark coordinates
y = df.iloc[:, -1]   # labels

# Encode labels (e.g., 'A' -> 0, 'B' -> 1, ...)
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Save processed data
X_train.to_csv(os.path.join(PROCESSED_DIR, 'X_train.csv'), index=False)
X_test.to_csv(os.path.join(PROCESSED_DIR, 'X_test.csv'), index=False)
pd.DataFrame(y_train).to_csv(os.path.join(PROCESSED_DIR, 'y_train.csv'), index=False)
pd.DataFrame(y_test).to_csv(os.path.join(PROCESSED_DIR, 'y_test.csv'), index=False)

# Save label mapping
label_map = pd.DataFrame({'label': le.classes_, 'encoded': range(len(le.classes_))})
label_map.to_csv(os.path.join(PROCESSED_DIR, 'label_map.csv'), index=False)

print("[✅] Preprocessing complete. Data saved in data/processed/")
