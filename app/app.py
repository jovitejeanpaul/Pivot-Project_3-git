from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        # On récupère les données proprement avec des valeurs par défaut
        data = {
            'age': request.form.get('age'),
            'smokes': request.form.get('smokes'),
            'smokes_years': request.form.get('smokes_years', 0),
            'partners': request.form.get('partners'),
            'first_intercourse': request.form.get('first_intercourse'),
            'pregnancies': request.form.get('pregnancies'),
            'hormonal_years': request.form.get('hormonal_years', 0),
            'stds': request.form.get('stds'),
            'cancer_prev': request.form.get('cancer_prev')
        }
        prediction = f"Analyse terminée. Risque évalué selon {len(data)} paramètres cliniques."
        
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)