

import cv2
import face_recognition
import numpy as np
import os
from datetime import datetime

# Load images and encode known faces
path = 'clp'
images = []
classNames = []
mylist = os.listdir(path)
print("Images in folder:", mylist)

for cl in mylist:
    curImg = cv2.imread(f'{path}/{cl}')
    if curImg is not None:
        images.append(curImg)
        classNames.append(os.path.splitext(cl)[0])
    else:
        print(f"Error: Unable to load image {cl}")

print("Class Names:", classNames)

def findEncodings(images):
    encodeList = []
    for i, img in enumerate(images):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(img)

        if encodings: 
            encodeList.append(encodings[0])
        else:
            print(f"Warning: No face found in {classNames[i]}")

    return encodeList

def markAttendance(name, status):
    now = datetime.now()
    dateString = now.strftime('%Y-%m-%d')  # Ensure correct format
    timeString = now.strftime('%H:%M:%S')
    
    with open('attendance.csv', 'r+') as f:
        myDataList = f.readlines()
        nameList = [line.split(',')[0] for line in myDataList]

        # Ensure each person is marked only once per day
        if name not in nameList:
            f.write(f'\n{name},{dateString},{timeString},{status}')
            print(f"Attendance Marked for {name} - {status}")

# Encode known faces
encodelistknown = findEncodings(images)
print('Encoding complete. Encoded faces:', len(encodelistknown))

# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    if not success:
        print("Error: Unable to access webcam")
        break

    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    faceCurFrame = face_recognition.face_locations(imgS)
    encodeCurFrame = face_recognition.face_encodings(imgS, faceCurFrame)

    for encodeFace, faceLoc in zip(encodeCurFrame, faceCurFrame):
        matches = face_recognition.compare_faces(encodelistknown, encodeFace)
        faceDis = face_recognition.face_distance(encodelistknown, encodeFace)

        matchIndex = np.argmin(faceDis) if len(faceDis) > 0 else -1
        if matchIndex != -1 and matches[matchIndex]:
            name = classNames[matchIndex].upper()
        else:
            name = "Unknown"

        y1, x2, y2, x1 = faceLoc
        y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4

        # Determine the attendance status
        now = datetime.now()
        hour, minute = now.hour, now.minute
        status = "Present"
        box_color = (0, 255, 0)  # Default color: Green for present
        
        # If the student is late
        if hour >= 12 and minute > 0:
            status = "Late"
            box_color = (0, 165, 255)  # Orange color for late students

        # If face is recognized
        if name != "Unknown":
            markAttendance(name, status)
            displayText = f"{name} - {status}"
        else:
            displayText = "Face Not Recognized"
            box_color = (0, 0, 255)  # Red for unrecognized faces

        # Draw rectangle and put text
        cv2.rectangle(img, (x1, y1), (x2, y2), box_color, 2)
        cv2.rectangle(img, (x1, y2 - 35), (x2, y2), box_color, cv2.FILLED)
        cv2.putText(img, displayText, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

    # Display current date and time on the webcam feed
    now = datetime.now()
    date_text = now.strftime('%Y-%m-%d %H:%M:%S')
    cv2.putText(img, date_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

    # Show the webcam feed
    cv2.imshow('Webcam', img)

    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break

cap.release()
cv2.destroyAllWindows()
