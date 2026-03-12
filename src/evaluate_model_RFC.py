from sklearn.metrics import (
    roc_auc_score, roc_curve,
    accuracy_score, precision_score,
    recall_score, f1_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
 
# --- 2.1 Calcul des métriques ---
roc_auc   = roc_auc_score(y_test, y_pred_proba)
accuracy  = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall    = recall_score(y_test, y_pred)
f1        = f1_score(y_test, y_pred)
 
print("=" * 48)
print("    PERFORMANCES SUR L'ENSEMBLE DE TEST (20%)")
print("=" * 48)
print(f"  ROC-AUC   : {roc_auc:.4f}")
print(f"  Accuracy  : {accuracy:.4f}")
print(f"  Precision : {precision:.4f}")
print(f"  Recall    : {recall:.4f}")
print(f"  F1-Score  : {f1:.4f}")
print("=" * 48)
 
# Rapport complet par classe
print("\nRapport de classification détaillé :")
print(classification_report(y_test, y_pred, target_names=["Pas de risque", "À risque"]))
 
# --- 2.2 Validation croisée (5 folds) pour confirmer la généralisation ---
cv_roc_auc = cross_val_score(modele, X_train_scaled, y_train, cv=5, scoring="roc_auc", n_jobs=-1)
print(f"ROC-AUC en validation croisée (5-fold) : {cv_roc_auc.mean():.4f} ± {cv_roc_auc.std():.4f}\n")
 
# --- 2.3 Visualisations ---
fig = plt.figure(figsize=(18, 5))
fig.suptitle("Évaluation du Random Forest — Risque de cancer du col de l'utérus", fontsize=13)
gs = gridspec.GridSpec(1, 3, figure=fig)
 
# Courbe ROC
ax1 = fig.add_subplot(gs[0])
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
ax1.plot(fpr, tpr, color="#e74c3c", lw=2, label=f"AUC = {roc_auc:.4f}")
ax1.plot([0, 1], [0, 1], "k--", lw=1)
ax1.set_xlabel("Taux de faux positifs (FPR)")
ax1.set_ylabel("Taux de vrais positifs (TPR)")
ax1.set_title("Courbe ROC")
ax1.legend(loc="lower right")
ax1.grid(alpha=0.3)
 
# Matrice de confusion
ax2 = fig.add_subplot(gs[1])
cm   = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Pas de risque", "À risque"])
disp.plot(cmap="Blues", ax=ax2, colorbar=False)
ax2.set_title("Matrice de Confusion")
 
# Top 15 variables importantes
ax3 = fig.add_subplot(gs[2])
feat_imp = pd.Series(modele.feature_importances_, index=X_train.columns if hasattr(X_train, 'columns') else range(X_train_scaled.shape[1]))
top15    = feat_imp.nlargest(15).sort_values()
ax3.barh(top15.index, top15.values, color="#3498db")
ax3.set_xlabel("Importance")
ax3.set_title("Top 15 — Variables les plus importantes")
ax3.grid(alpha=0.3, axis="x")
 
plt.tight_layout()
plt.savefig("evaluation_random_forest.png", dpi=150, bbox_inches="tight")
plt.show()
print("📊 Graphiques sauvegardés : evaluation_random_forest.png")
 
# --- 2.4 Tableau récapitulatif exportable ---
resultats = pd.DataFrame({
    "Métrique": ["ROC-AUC", "Accuracy", "Precision", "Recall", "F1-Score"],
    "Score"   : [roc_auc, accuracy, precision, recall, f1]
}).round(4)
 
print("\nTableau récapitulatif :")
print(resultats.to_string(index=False))
resultats.to_csv("resultats_random_forest.csv", index=False)
print("💾 Résultats exportés : resultats_random_forest.csv")