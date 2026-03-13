from flask import Flask, render_template, request
import pickle
import numpy as np
# Charger le modèle Random Forest déjà présent dans votre dossier src
model_path = 'src/random_forest_model.pkl'
with open(model_path, 'rb') as f:
    model = pickle.load(f)
app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None  # On garde cette ligne pour initialiser la variable

    if request.method == 'POST':
        # 1. On récupère les données
        try:
            data = [
                float(request.form.get('age', 0)),
                float(request.form.get('smokes', 0)),
                float(request.form.get('smokes_years', 0)),
                float(request.form.get('partners', 0)),
                float(request.form.get('first_intercourse', 0)),
                float(request.form.get('pregnancies', 0)),
                float(request.form.get('hormonal_years', 0)),
                float(request.form.get('stds', 0)),
                float(request.form.get('cancer_prev', 0))
            ]

            # 2. Transformation pour l'algorithme
            features = np.array([data])

            # 3. Prédiction avec le modèle chargé
            prediction_value = model.predict(features)[0]
            
            # 4. On écrase le 'None' par le vrai message
            if prediction_value == 1:
                prediction = "Risque élevé détecté. Veuillez consulter un spécialiste."
            else:
                prediction = "Faible risque détecté."
        
        except Exception as e:
            prediction = f"Erreur lors du calcul : {str(e)}"

    # On renvoie toujours 'prediction' au template
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)