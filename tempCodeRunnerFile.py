

@app.route('/predict',methods=['POST'])
def predict():   
    req = request.get_json()