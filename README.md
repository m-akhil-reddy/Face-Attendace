# Face Recognition Attendance System

## Overview
This project implements a real-time face recognition-based attendance system using Python. It utilizes computer vision libraries to detect and recognize faces via a webcam, marking attendance in a CSV file. The system distinguishes between "Present" and "Late" statuses based on the current time.

## Features
- Real-time face detection and recognition using webcam
- Automatic attendance marking in CSV format (like :- MS excel)
- Time-based status determination (Present/Late)
- Visual feedback with colored bounding boxes
- Handles unknown faces gracefully

## Requirements
- Python 3.x
- Webcam (built-in or external)
- Required Python libraries:
  - `opencv-python` (cv2)
  - `face-recognition`
  - `numpy`

## Installation
1. Clone or download the project files.
2. Install the required dependencies:
   ```
   pip install opencv-python face-recognition numpy
   ```
3. Ensure you have images of individuals in the `clp/` folder (e.g., `roll-no-1.jpg`, `roll-no-2.jpg`). The filename (without extension) will be used as the person's roll.no or name.

## Usage
1. Place face images in the `clp/` folder.
2. Run the script:
   ```
   python project.py
   ```
3. The webcam will activate. Recognized faces will be marked in `attendance.csv`.
4. Press 'q' to quit the application.

## Folder Structure
- `project.py`: Main Python script
- `clp/`: Folder containing face images for recognition
- `attendance.csv`: Output file for attendance records

## How It Works
- Loads and encodes faces from images in `clp/`
- Captures webcam feed and detects faces
- Compares detected faces with known encodings
- Marks attendance with timestamp and status
- Displays real-time feedback on the video feed

## Notes
- Ensure good lighting for accurate face recognition.
- The system considers attendance after 12:00 PM as "Late".
- Unknown faces are not marked in the attendance file.
- Make sure your webcam is accessible and not in use by other applications.

## Troubleshooting
- If faces are not recognized, try adjusting lighting or image quality.
- Ensure all dependencies are installed correctly.
- Check webcam permissions if the script fails to access the camera.
