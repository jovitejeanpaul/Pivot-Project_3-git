from flask import Flask, render_template, request
import numpy as np
import os
import shap
import matplotlib
matplotlib.use('Agg')  # IMPORTANT : pas d'écran, rendu fichier
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier
import joblib
import pandas as pd


app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, '..', 'modele_cancer_final.cbm')

model = CatBoostClassifier()
model.load_model(model_path)

# ← AJOUTER CES 2 LIGNES
scaler   = joblib.load(os.path.join(BASE_DIR, '..', 'modele_scaler.pkl'))
colonnes = joblib.load(os.path.join(BASE_DIR, '..', 'modele_colonnes.pkl'))
FEATURE_NAMES = [
    "Age",
    "Number of sexual partners",
    "First sexual intercourse",
    "Num of pregnancies",
    "Smokes",
    "Smokes (years)",
    "Smokes (packs/year)",
    "Hormonal Contraceptives",
    "Hormonal Contraceptives (years)",
    "IUD",
    "IUD (years)",
    "STDs",
    "STDs:cervical condylomatosis",
    "STDs:vaginal condylomatosis",
    "STDs:syphilis",
    "STDs:pelvic inflammatory disease",
    "STDs:genital herpes",
    "STDs:molluscum contagiosum",
    "STDs:AIDS",
    "STDs:HIV",
    "STDs:Hepatitis B",
    "STDs:HPV",
    "Dx:Cancer",
    "Dx:CIN",
    "Dx",
    "Hinselmann",
    "Schiller",
    "Citology",
]

def generate_shap_chart(features_array, save_path):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(features_array)

    # Compatibilité CatBoost binaire
    if isinstance(shap_values, list):
        sv = shap_values[1][0]
    elif shap_values.ndim == 3:
        sv = shap_values[0, :, 1]
    else:
        sv = shap_values[0]

    # Top 15 features par impact absolu
    indices = np.argsort(np.abs(sv))[::-1][:15]
    top_names  = [FEATURE_NAMES[i] for i in indices]
    top_values = sv[indices]
    colors = ['#e74c3c' if v > 0 else '#3498db' for v in top_values]

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor('#fdf0f4')
    ax.set_facecolor('#fdf0f4')

    ax.barh(range(len(top_values)), top_values[::-1],
            color=colors[::-1], edgecolor='white', height=0.6)
    ax.set_yticks(range(len(top_values)))
    ax.set_yticklabels(top_names[::-1], fontsize=10)
    ax.axvline(0, color='#333', linewidth=0.8, linestyle='--')
    ax.set_xlabel("Impact sur la prédiction (valeur SHAP)", fontsize=10)
    ax.set_title("Facteurs ayant influencé le diagnostic", fontsize=12,
                 fontweight='bold', color='#007bff', pad=12)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#e74c3c', label='↑ Augmente le risque'),
        Patch(facecolor='#3498db', label='↓ Diminue le risque'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9,
              framealpha=0.7, facecolor='white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()


@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    risk_level  = None
    shap_image  = None
    show_result = False
    risk_percent = 0

    if request.method == 'POST':
        show_result = True
        try:
            features = [
                float(request.form.get('age', 0)),
                float(request.form.get('partners', 0)),
                float(request.form.get('first_intercourse', 0)),
                float(request.form.get('pregnancies', 0)),
                float(request.form.get('smokes', 0)),
                float(request.form.get('smokes_years', 0)),
                float(request.form.get('smokes_packs', 0)),
                float(request.form.get('hormonal', 0)),
                float(request.form.get('hormonal_years', 0)),
                float(request.form.get('iud', 0)),
                float(request.form.get('iud_years', 0)),
                float(request.form.get('stds', 0)),
                float(request.form.get('stds_cervical', 0)),
                float(request.form.get('stds_vaginal', 0)),
                float(request.form.get('stds_syphilis', 0)),
                float(request.form.get('stds_pid', 0)),
                float(request.form.get('stds_herpes', 0)),
                float(request.form.get('stds_molluscum', 0)),
                float(request.form.get('stds_aids', 0)),
                float(request.form.get('stds_hiv', 0)),
                float(request.form.get('stds_hepb', 0)),
                float(request.form.get('stds_hpv', 0)),
                float(request.form.get('dx_cancer', 0)),
                float(request.form.get('dx_cin', 0)),
                float(request.form.get('dx', 0)),
                float(request.form.get('hinselmann', 0)),
                float(request.form.get('schiller', 0)),
                float(request.form.get('citology', 0)),
            ]

            X_df     = pd.DataFrame([features], columns=FEATURE_NAMES)
            X_df     = X_df[colonnes]           # filtre les bonnes colonnes
            X_scaled = scaler.transform(X_df)   # normalise comme à l'entraînement
            
            result = model.predict(X_scaled)[0]
            proba        = model.predict_proba(X_scaled)[0][1]  # ← AJOUTER
            risk_percent = round(proba * 100, 1)                # ← AJOUTER

            if result == 1:
                prediction = "Risque élevé détecté."
                risk_level  = "high"
            else:
                prediction = "Faible risque détecté."
                risk_level  = "low"

            # Génération SHAP → static/shap_result.png
            shap_filename = 'shap_result.png'
            shap_path = os.path.join(BASE_DIR, 'static', shap_filename)
            generate_shap_chart(X_scaled, shap_path)
            shap_image = shap_filename

        except Exception as e:
            prediction = f"Erreur : {str(e)}"

    return render_template('index.html',
                           prediction=prediction,
                           risk_level=risk_level,
                           shap_image=shap_image,
                           show_result = True,
                           risk_percent=risk_percent)


if __name__ == '__main__':
    app.run(debug=True)
