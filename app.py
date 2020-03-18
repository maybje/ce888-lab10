from flask import Flask, request, render_template
import os
import pickle

print(os.getcwd())
path = os.getcwd()

with open('Models/logistic_heart.pkl', 'rb') as f:
    logistic = pickle.load(f)

with open('Models/rf_heart.pkl', 'rb') as f:
    randomforest = pickle.load(f)

with open('Models/svm_heart.pkl', 'rb') as f:
    svm_model = pickle.load(f)


def get_predictions(age, sex, cp, trestbps, chol, fbs, restcg, thalach, exang, oldpeak, slope, ca, thal, req_model):
    mylist = [age, sex, cp, trestbps, chol, fbs, restcg, thalach, exang, oldpeak, slope, ca, thal]
    mylist = [float(i) for i in mylist]
    vals = [mylist]

    if req_model == 'Logistic':
        #print(req_model)
        return logistic.predict(vals)[0]

    elif req_model == 'RandomForest':
        #print(req_model)
        return randomforest.predict(vals)[0]

    elif req_model == 'SVM':
        #print(req_model)
        return svm_model.predict(vals)[0]
    else:
        return "Cannot Predict"


app = Flask(__name__)

@app.route('/')
def my_form():
    return render_template('home.html')

@app.route('/', methods=['POST', 'GET'])
def my_form_post():
    age = request.form['age']
    sex = request.form['sex']
    cp = request.form['cp']
    trestbps = request.form['trestbps']
    chol = request.form['chol']
    fbs = request.form['fbs']
    restcg = request.form['restcg']
    thalach = request.form['thalach']
    exang = request.form['exang']
    oldpeak = request.form['oldpeak']
    slope = request.form['slope']
    ca = request.form['ca']
    thal = request.form['thal']
    req_model = request.form['req_model']

    target = get_predictions(age, sex, cp, trestbps, chol, fbs, restcg, thalach, exang, oldpeak, slope, ca, thal, req_model)

    if target==1:
        heart_disease = 'Patient has a heart disease'
    else:
        heart_disease = 'Patient does not have a heart disease'

    return render_template('home.html', target = target, heart_disease = heart_disease)


if __name__ == "__main__":
    app.run(debug=True)
