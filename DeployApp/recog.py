from PIL import Image, ImageDraw
import face_recognition
import numpy as np
import pickle
import joblib
import os
import pickle

# Dependencies
# from flask import Flask, request, jsonify,make_response
# import urllib.request


class FaceRecognize:
    def fit(self,Path):
        PathList = list()
        self.known_face_encodings = list()
        self.known_face_names = list()
        
        # Get all subdirectories
        FolderList = os.listdir(Path)
        print("Training starts")
        print(len(FolderList))
        # Loop over each directory
        for File in FolderList:
            if(os.path.isdir(Path + os.path.sep + File)):
                for Image in os.listdir(Path + os.path.sep + File):               
                    # Add the image path to the list
                    PathList.append(Path + os.path.sep + File + os.path.sep + Image)
                    
                    # Add a label for each image and remove the file extension      
                    self.known_face_names.append(File.split(".")[0])
            
            else:
                PathList.append(Path + os.path.sep + File)
                
                # Add a label for each image and remove the file extension
                self.known_face_names.append(File.split(".")[0])
        j = 0
        for (i,imagepath) in enumerate(PathList):
            try:
                image = face_recognition.load_image_file(imagepath)
                face_encoding = face_recognition.face_encodings(image)[0]
                self.known_face_encodings.append(face_encoding)
                
            except IndexError as error:
                print(i)
                del self.known_face_names[i]
                print(self.known_face_names[i])
                print(error)
                pass           
            
        print(len(self.known_face_names))
        print(len(self.known_face_encodings))
        print(j)
        
    def predict(self,PredictPath,temp):
        unknown_image = face_recognition.load_image_file(PredictPath)
        
        # Find all the faces and face encodings in the unknown image
        face_locations = face_recognition.face_locations(unknown_image)
        face_encodings = face_recognition.face_encodings(unknown_image, face_locations)
        
        pil_image = Image.fromarray(unknown_image)
        
        for face_encoding in  face_encodings:
            
            # See if the face is a match for the known face(s)       
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
            name = "Unknown"
            
            # Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = self.known_face_names[best_match_index]
                name = self.known_face_names[best_match_index]
            
        # return temp==name
        return name==temp
    




# model = pickle.load(open('hello.pkl','rb'))
# # Your API definition
# app = Flask(__name__)


# @app.route('/')
# def man():
#     return {'message': 'Hello World'}

# @app.route('/predict',methods=['POST'])
# def predict():   
#     req = request.get_json()
#     filename = 'face.jpg'
#     #image_url = "https://thumbs.dreamstime.com/z/happy-little-boy-smiley-face-portrait-human-concept-freshness-133726078.jpg"
#     image_url = str(req['URL'])
#     urllib.request.urlretrieve(image_url,filename)
#     print(req)
#     pred = model.predict(filename,req['ROLLNO'])
#     check = {"message" :str(pred)}
#     print(check)
#     return check 




# # if __name__ == '__main__':
# #     app.run(debug=True)

# if __name__ == '__main__':
#     Path = os.path.join(os.getcwd(), 'test')
#     model = FaceRecognize()
#     model.fit(Path)
#     print(Path)
    # PredictPath = "E:\College\Fifth Semester\Software Engineering\Project\Face Recognize\Images\Predict\check4.jpg"
    # print("Enter the name")
    # RollNo = "1810110088"
    # # Current Model
    # print(model.predict(PredictPath,RollNo))
    # # Write the model on the disk using pickle
    # pickle.dump(model,open('hello.pkl','wb'))
    
    # check = pickle.load(open('hello.pkl','rb'))
    # print(check.predict(PredictPath,RollNo))

