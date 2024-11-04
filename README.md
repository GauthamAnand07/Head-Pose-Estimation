# Head Pose Estimation

Head Pose Detection using MediaPipe Face Mesh and OpenCV  
This project detects head poses (looking left, right, up, down, or forward) in real-time using facial landmarks with the MediaPipe Face Mesh solution and OpenCV. It uses the webcam feed to capture frames and displays the detected head orientation on the screen.

## Table of Contents
- [Demo](#demo)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Code Overview](#code-overview)

---

## Demo

A sample of head pose detection.

---

## Features

- Real-time head pose detection using webcam.
- Determines head direction (left, right, up, down, forward).
- Displays orientation angle information on the screen.
- Uses MediaPipe's Face Mesh model for robust landmark detection.

---

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/GauthamAnand07/Head-Pose-Estimation.git

2. **Install Dependencies:**
   ```bash
   pip install opencv-python mediapipe numpy
   
## Usage

   **Run the following command to start the head pose detection:**
   ```bash
   python pose_estimation.py.py
   ```
Press the Esc key to exit the application.

## Code Overview
1. **Import Libraries**
The code uses:
 ```bash
import cv2
import mediapipe as mp
import numpy as np
 ```

2. **Initialize Face Mesh Model**
The Face Mesh model is initialized with a confidence threshold of 0.5 for reliable landmark detection.
 ```bash
   mp_face_mesh = mp.solutions.face_mesh
   face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)
   ```

3. **Define Drawing Specifications**
Sets the landmark color, thickness, and radius for displaying detected facial landmarks.
 ```bash
  drawing_spec = mp.solutions.drawing_utils.DrawingSpec(thickness=1, circle_radius=1)
 ```
4. **Video Capture and Processing**
Captures frames from the webcam and processes each frame:

 ```bash
cap = cv2.VideoCapture(0)
while cap.isOpened():
    success, frame = cap.read()
    frame = cv2.flip(frame, 1)  # Mirror view
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
 ```
5. **Camera Matrix and Distortion Coefficients**
Sets up a camera matrix for mapping 3D points to the 2D image plane. Distortion coefficients are set to zero, assuming no lens distortion.

 ```bash
focal_length = 1
cam_matrix = np.array([[focal_length, 0, frame.shape[1] / 2],
                       [0, focal_length, frame.shape[0] / 2],
                       [0, 0, 1]])
dist_coeffs = np.zeros((4, 1))
 ```
6. **Head Pose Estimation**
Calculates the rotation and translation vectors using OpenCV’s solvePnP. Converts the rotation vector to a rotation matrix and decomposes it into Euler angles to get the head's orientation.

 ```bash
  success, rotation_vector, translation_vector = cv2.solvePnP(model_points, image_points, cam_matrix, dist_coeffs)
 ```

7. **Direction Determination**
Based on the calculated angles:

Negative y angle -> Looking left
Positive y angle -> Looking right
Negative x angle -> Looking down
Positive x angle -> Looking up
No conditions met -> Looking forward
   
