
#%%

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import numpy as np
import data_processing as dp

data_train = dp.X_train 
# Remplacement du modèle par défaut par une version avec hyperparamètres optimisés
modele = RandomForestClassifier(
    n_estimators=200,        # nombre d'arbres
    max_depth=None,          # profondeur libre (les arbres poussent jusqu'au bout)
    min_samples_split=5,     # nb min d'échantillons pour diviser un nœud
    min_samples_leaf=2,      # nb min d'échantillons dans une feuille
    max_features="sqrt",     # nb de features considérées à chaque split
    class_weight="balanced", # compense le déséquilibre des classes (important en médecine)
    random_state=42,
    n_jobs=-1                # utilise tous les cœurs disponibles
)
 
modele.fit(data_train, y_train)
print("✅ Modèle Random Forest entraîné avec succès.\n")
 
# Prédictions sur le jeu de test
y_pred       = modele.predict(X_test_scaled)
y_pred_proba = modele.predict_proba(X_test_scaled)[:, 1]  # probabilités pour la classe positive
 