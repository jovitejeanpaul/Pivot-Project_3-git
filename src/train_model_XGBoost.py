import joblib
from xgboost import XGBClassifier

# --- ETAPE D'ENTRAINEMENT ---
# X_train_final, y_train_balanced sont tes jeux de données

def train_and_save(X_train, y_train):
    # Initialisation du modèle XGBoost
    # Note : Ton ami a déjà utilisé SMOTE, donc pas besoin de scale_pos_weight ici
    model = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    
    # Entraînement
    model.fit(X_train, y_train)
    
    # Sauvegarde du modèle
    joblib.dump(model, 'cancer_model_xgb.pkl')
    
    # Sauvegarde du scaler (important pour l'évaluation !)
    # Si ton ami a nommé son scaler 'scaler' :
    # joblib.dump(scaler, 'scaler.pkl') 
    
    print("Modèle entraîné et sauvegardé avec succès.")
    return model

# Appel de la fonction (à adapter selon ton intégration)
# train_and_save(X_train_final, y_train_balanced)