import shap
import pandas as pd

def shap_explainability(modele, X_patient):
    """
    Génère l'explicabilité SHAP pour les modèles basés sur des arbres 
    (Random Forest, XGBoost, CatBoost) en gérant leurs différences de format.
    
    Paramètres:
    - modele : Le modèle d'apprentissage automatique déjà entraîné.
    - X_patient : Un DataFrame (1 ligne) contenant les données prétraitées de la patiente.
    
    Retourne:
    - explainer : L'objet TreeExplainer de SHAP.
    - shap_values_cible : Les valeurs SHAP associées uniquement à la classe "À risque" (1).
    """
    # Initialisation de l'explicateur SHAP spécifique aux arbres
    explainer = shap.TreeExplainer(modele)
    
    # Calcul des valeurs SHAP pour la patiente
    shap_values = explainer.shap_values(X_patient)
    
    # Gestion de la différence de format de sortie entre les bibliothèques
    if isinstance(shap_values, list):
        # Pour Random Forest (Scikit-Learn) : on récupère les valeurs de la classe 1
        shap_values_cible = shap_values[1]
    else:
        # Pour XGBoost et CatBoost : le format est déjà correct pour la classe positive
        shap_values_cible = shap_values
        
    return explainer, shap_values_cible