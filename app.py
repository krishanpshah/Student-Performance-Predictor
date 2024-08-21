import numpy as np
from flask import Flask, request, render_template
import pickle
import os


model = pickle.load(open('model.pkl','rb'))
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=["POST"])
def predict():
    
    init_features=[int(x) for x in request.form.values()]
    final_features=[np.array(init_features)]
    prediction=model.predict(final_features)
    output=round(prediction[0],1)
    if output <=0:
        output=0
    elif output >=100:
        output=100
    return render_template('index.html', prediction_text='Your predicted test score is: {}'.format(output))



if __name__=='__main__':
    app.run(debug=True,host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))