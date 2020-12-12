# Dependencies
from flask import Flask, request, jsonify,make_response
from flask_cors import CORS, cross_origin
import pickle
import urllib.request
from recog import FaceRecognize

model = pickle.load(open('hello.pkl','rb'))
# Your API definition
app = Flask(__name__)
CORS(app)


# API DETAILS :
# POST REQUEST, BODY DETAILS :
# {
#     "URL": "URL",
#     "ROLLNO": "ROLLNO"
# }

@app.route('/predict',methods=['POST'])
@cross_origin()
def predict():  
    req = request.get_json()
    filename = 'face.jpg'
    image_url = str(req['URL'])
    urllib.request.urlretrieve(image_url,filename)
    print(req)
    pred = model.predict(filename,req['ROLLNO'])
    check = {"message" :str(pred)}
    print(check)
    return check        


if __name__ == '__main__':
    app.run(debug=True)

# app.run(debug=True)