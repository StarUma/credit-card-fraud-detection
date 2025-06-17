import pandas as pd
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

def load_data(filepath):
    return pd.read_csv(filepath)

def preprocess_data(df):
    scaler = StandardScaler()
    df['scaled_amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
    df['scaled_time'] = scaler.fit_transform(df['Time'].values.reshape(-1, 1))
    df = df.drop(['Time', 'Amount'], axis=1)

    X = df.drop('Class', axis=1)
    y = df['Class']

    sm = SMOTE(random_state=42)
    X_resampled, y_resampled = sm.fit_resample(X, y)

    return X_resampled, y_resampled
