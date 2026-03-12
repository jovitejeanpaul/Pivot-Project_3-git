import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def evaluate_saved_model(X_test, y_test):
    # 1. Chargement du modèle
    model = joblib.load('cancer_model_xgb.pkl')
    
    # 2. Prédictions
    y_pred = model.predict(X_test)
    
    # 3. Rapport de performance
    print("--- RAPPORT DE CLASSIFICATION ---")
    print(classification_report(y_test, y_pred))
    
    # 4. Matrice de confusion
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds')
    plt.title('Matrice de Confusion - Test Set')
    plt.xlabel('Prédiction')
    plt.ylabel('Réalité')
    plt.show()

# Appel de la fonction
# evaluate_saved_model(X_test_final, y_test)