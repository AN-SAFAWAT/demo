from flask import Flask, request, jsonify, render_template 

import numpy as np
import pickle


app = Flask(__name__)

model=pickle.load(open('mobile_pursc.pkl','rb'))
mobile_sc=pickle.load(open('scaler.pickle','rb'))

                         
@app.route('/')

def home ():
    return render_template('index.html')

@app.route ('/predict',methods=['POST'])

def predict():
   
    ##For rendering results on HTML GUI
    
    
    int_features = [int(x) for x in request.form.values()]
    pre_final_features =[np.array(int_features)]
    final_features = mobile_sc.transform(pre_final_features)
    prediction = model.predict(final_features)
    #('predictio value is ',prediction[0])
    if (prediction[0]== 1):
        
        output = "True"
    elif(prediction[0] == 0):
        
        output = "False"
    else:
        output = "Not sure"
        
        
    return render_template('index.html', prediction_text='This user will buy mobile: state {}'.format(output))

if __name__ == "__main__":
     app.run(host='0.0.0.0', port=8080)
