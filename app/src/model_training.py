import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from src.data_preprocessing import load_data, preprocess_data

def train_model():
    df = load_data('data/creditcard.csv')
    X, y = preprocess_data(df)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(class_weight='balanced')
    model.fit(X_train, y_train)  

    # Save model
    with open('models/fraud_detection_model.pkl', 'wb') as f:
        pickle.dump(model, f)

    # Evaluate
    y_pred = model.predict(X_test)
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("ROC AUC Score:", roc_auc_score(y_test, y_pred))

if __name__ == "__main__":
    train_model()
