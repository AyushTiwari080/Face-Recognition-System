import cv2
import os
from flask import Flask, render_template, redirect, request, jsonify, Response
from datetime import date, datetime
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import joblib

app = Flask(__name__)

# Initialize variables
datetoday = date.today().strftime("%m_%d_%y")
datetoday2 = date.today().strftime("%d-%B-%Y")

# Initialize face detector
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Create required directories
os.makedirs('Attendance', exist_ok=True)
os.makedirs('static/faces', exist_ok=True)

attendance_file = f'Attendance/Attendance-{datetoday}.csv'
if not os.path.exists(attendance_file):
    with open(attendance_file, 'w') as f:
        f.write('Name,Roll,Time\n')

def totalreg():
    return len(os.listdir('static/faces'))

def extract_faces(img):
    if img is not None:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_points = face_detector.detectMultiScale(gray, 1.3, 5)
        return face_points
    return []

def identify_face(facearray):
    model = joblib.load('static/face_recognition_model.pkl')
    return model.predict(facearray)

def train_model():
    faces, labels = [], []
    userlist = os.listdir('static/faces')
    for user in userlist:
        for imgname in os.listdir(f'static/faces/{user}'):
            img = cv2.imread(f'static/faces/{user}/{imgname}')
            if img is None:
                continue
            resized_face = cv2.resize(img, (50, 50))
            faces.append(resized_face.ravel())
            labels.append(user)
    if len(faces) > 0:
        faces = np.array(faces)
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(faces, labels)
        joblib.dump(knn, 'static/face_recognition_model.pkl')

def extract_attendance():
    df = pd.read_csv(attendance_file)
    return df['Name'], df['Roll'], df['Time'], len(df)

def add_attendance(name):
    username, userid = name.split('_')
    current_time = datetime.now().strftime("%H:%M:%S")
    df = pd.read_csv(attendance_file)
    if int(userid) not in list(df['Roll']):
        with open(attendance_file, 'a') as f:
            f.write(f'{username},{userid},{current_time}\n')

@app.route('/')
def home():
    names, rolls, times, l = extract_attendance()
    return render_template('home.html', 
                           names=names, 
                           rolls=rolls, 
                           times=times, 
                           l=l, 
                           totalreg=totalreg(), 
                           datetoday2=datetoday2)

@app.route('/start', methods=['GET'])
def start():
    if 'face_recognition_model.pkl' not in os.listdir('static'):
        return render_template('home.html', 
                               totalreg=totalreg(), 
                               datetoday2=datetoday2, 
                               mess='No trained model found. Please add a new face.')

    # Use browser-based attendance page
    return render_template('attendance.html', 
                          totalreg=totalreg(), 
                          datetoday2=datetoday2)

@app.route('/video_feed')
def video_feed():
    """Video streaming route."""
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def gen_frames():
    """Video generator function."""
    camera = cv2.VideoCapture(0)
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            faces = extract_faces(frame)
            if len(faces) > 0:
                (x, y, w, h) = faces[0]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 20), 2)
                face = cv2.resize(frame[y:y + h, x:x + w], (50, 50))
                identified_person = identify_face(face.reshape(1, -1))[0]
                add_attendance(identified_person)
                cv2.putText(frame, f'{identified_person}', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2, cv2.LINE_AA)
            
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/add', methods=['GET', 'POST'])
def add():
    if request.method == 'GET':
        # Show capture page with browser webcam
        return render_template('capture.html', 
                              totalreg=totalreg(), 
                              datetoday2=datetoday2)
    
    # POST request - receive captured images from browser
    newusername = request.form.get('newusername')
    newuserid = request.form.get('newuserid')
    image_data = request.form.get('image')
    
    if not newusername or not newuserid:
        names, rolls, times, l = extract_attendance()
        return render_template('home.html', 
                               names=names, 
                               rolls=rolls, 
                               times=times, 
                               l=l, 
                               totalreg=totalreg(), 
                               datetoday2=datetoday2,
                               mess='Please provide both name and ID')
    
    userimagefolder = f'static/faces/{newusername}_{newuserid}'
    os.makedirs(userimagefolder, exist_ok=True)
    
    # Save image if provided
    if image_data:
        import base64
        # Remove data URL prefix if present
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        img_bytes = base64.b64decode(image_data)
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        faces = extract_faces(img)
        if len(faces) > 0:
            (x, y, w, h) = faces[0]
            face_img = img[y:y + h, x:x + w]
            
            # Count existing images
            existing = len([f for f in os.listdir(userimagefolder) if f.endswith('.jpg')])
            name = f'{newusername}_{existing + 1}.jpg'
            cv2.imwrite(f'{userimagefolder}/{name}', face_img)
            
            return jsonify({'success': True, 'count': existing + 1})
    
    # Train the model after receiving images
    train_model()
    
    names, rolls, times, l = extract_attendance()
    return render_template('home.html', 
                           names=names, 
                           rolls=rolls, 
                           times=times, 
                           l=l, 
                           totalreg=totalreg(), 
                           datetoday2=datetoday2,
                           mess=f'Successfully added {newusername}!')

@app.route('/complete_capture', methods=['POST'])
def complete_capture():
    """Complete the capture process and train the model."""
    newusername = request.form.get('newusername')
    newuserid = request.form.get('newuserid')
    
    # Train the model
    train_model()
    
    names, rolls, times, l = extract_attendance()
    return render_template('home.html', 
                           names=names, 
                           rolls=rolls, 
                           times=times, 
                           l=l, 
                           totalreg=totalreg(), 
                           datetoday2=datetoday2,
                           mess=f'Successfully added {newusername}!')

@app.route('/detect_face', methods=['POST'])
def detect_face():
    """Detect face in the uploaded image."""
    data = request.get_json()
    image_data = data.get('image', '')
    
    if not image_data:
        return jsonify({'face_detected': False})
    
    try:
        import base64
        # Remove data URL prefix if present
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        img_bytes = base64.b64decode(image_data)
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        faces = extract_faces(img)
        
        if len(faces) > 0:
            (x, y, w, h) = faces[0]
            face = cv2.resize(img[y:y + h, x:x + w], (50, 50))
            identified_person = identify_face(face.reshape(1, -1))[0]
            add_attendance(identified_person)
            
            return jsonify({
                'face_detected': True,
                'name': identified_person,
                'x': int(x),
                'y': int(y),
                'w': int(w),
                'h': int(h)
            })
    except Exception as e:
        print(f"Error in detect_face: {e}")
    
    return jsonify({'face_detected': False})

@app.route('/get_attendance', methods=['GET'])
def get_attendance():
    """Get current attendance data."""
    names, rolls, times, l = extract_attendance()
    attendance = []
    for i in range(l):
        attendance.append({
            'name': str(names.iloc[i]) if i < len(names) else '',
            'roll': int(rolls.iloc[i]) if i < len(rolls) else '',
            'time': str(times.iloc[i]) if i < len(times) else ''
        })
    return jsonify({'attendance': attendance})

if __name__ == '__main__':
    app.run(debug=True)

