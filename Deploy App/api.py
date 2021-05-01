# Dependencies
from flask import Flask, request, jsonify,make_response
import pickle
import urllib.request
# from recog import FaceRecognize
import os
recog = __import__('recog')
FaceRecognize = recog.FaceRecognize

model = pickle.load(open('hello.pkl','rb'))
# Your API definition
app = Flask(__name__)

@app.route('/')
def man():
    return {'message': 'Hello World'}

@app.route('/predict',methods=['POST'])
def predict():   
    req = request.get_json()
    filename = 'face.jpg'
    #image_url = "https://thumbs.dreamstime.com/z/happy-little-boy-smiley-face-portrait-human-concept-freshness-133726078.jpg"
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