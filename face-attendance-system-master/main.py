import os.path
import datetime
import pickle

from fastapi import FastAPI, HTTPException
import cv2
import face_recognition
from PIL import Image

import util
from face_test import test

app = FastAPI()

class App:
    def __init__(self):
        self.db_dir = './db'
        if not os.path.exists(self.db_dir):
            os.mkdir(self.db_dir)

        self.log_path = './log.txt'
        self.cap = cv2.VideoCapture(0)

    def process_webcam(self):
        ret, frame = self.cap.read()
        self.most_recent_capture_arr = frame

    def login(self, name: str):
        self.process_webcam()

        label = test(
            image=self.most_recent_capture_arr,
            model_dir=r'E:\git-fashion\face-attendance-system-master\Silent-Face-Anti-Spoofing-master\resources\anti_spoof_models',
            device_id=0
        )

        if label == 1:
            name = util.recognize(self.most_recent_capture_arr, self.db_dir)

            if name in ['unknown_person', 'no_persons_found']:
                raise HTTPException(status_code=404, detail="Unknown user. Please register new user or try again.")
            else:
                with open(self.log_path, 'a') as f:
                    f.write('{},{},in\n'.format(name, datetime.datetime.now()))

                return {"message": f"Welcome, {name}."}
        else:
            raise HTTPException(status_code=400, detail="You are fake!")

    def logout(self, name: str):
        self.process_webcam()

        label = test(
            image=self.most_recent_capture_arr,
            model_dir=r'E:\git-fashion\face-attendance-system-master\Silent-Face-Anti-Spoofing-master\resources\anti_spoof_models',
            device_id=0
        )

        if label == 1:
            name = util.recognize(self.most_recent_capture_arr, self.db_dir)

            if name in ['unknown_person', 'no_persons_found']:
                raise HTTPException(status_code=404, detail="Unknown user. Please register new user or try again.")
            else:
                with open(self.log_path, 'a') as f:
                    f.write('{},{},out\n'.format(name, datetime.datetime.now()))

                return {"message": f"Goodbye, {name}."}
        else:
            raise HTTPException(status_code=400, detail="You are fake!")

    def register_new_user(self, name: str):
        self.process_webcam()

        # Save the image as the user's profile picture
        img_ = cv2.cvtColor(self.most_recent_capture_arr, cv2.COLOR_BGR2RGB)
        most_recent_capture_pil = Image.fromarray(img_)
        most_recent_capture_pil.save(os.path.join(self.db_dir, f'{name}.jpg'))

        embeddings = face_recognition.face_encodings(self.most_recent_capture_arr)[0]

        file = open(os.path.join(self.db_dir, f'{name}.pickle'), 'wb')
        pickle.dump(embeddings, file)

        return {"message": "User was registered successfully!"}

app_instance = App()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/login")
def login(name: str):
    return app_instance.login(name)

@app.post("/logout")
def logout(name: str):
    return app_instance.logout(name)

@app.post("/register")
def register_new_user(name: str):
    return app_instance.register_new_user(name)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)
