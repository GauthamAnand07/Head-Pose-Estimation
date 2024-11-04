# Head-Pose-Eastimation

Head Pose Detection using MediaPipe Face Mesh and OpenCV
This project detects head poses (looking left, right, up, down, or forward) in real-time using facial landmarks with the MediaPipe Face Mesh solution and OpenCV. It uses the webcam feed to capture frames and displays the detected head orientation on the screen.

Table of Contents
Demo
Features
Installation
Usage
Code Overview
Troubleshooting
License


Demo

A sample of head pose detection.




Features

Real-time head pose detection using webcam.
Determines head direction (left, right, up, down, forward).
Displays orientation angle information on the screen.
Uses MediaPipe's Face Mesh model for robust landmark detection.


Installation

Clone the repository:
git clone https://github.com/GauthamAnand07/Head-Pose-Estimation.git
cd head-pose-detection

Install dependencies:
pip install opencv-python mediapipe numpy

Usage

Run the following command to start the head pose detection:
python head_pose_detection.py
Press the Esc key to exit the application.

Code Overview

1. Import Libraries
The code uses:
OpenCV for video capture and display,
MediaPipe for face landmark detection, and
NumPy for numerical operations.

2. Initialize Face Mesh Model
The Face Mesh model is initialized with a confidence threshold of 0.5 for reliable landmark detection.

3. Define Drawing Specifications
Set the landmark color, thickness, and radius for displaying detected facial landmarks.

4. Video Capture and Processing
Captures frames from the webcam and processes each frame:
Flips each frame horizontally for a mirror view.
Converts it to RGB for MediaPipe processing.
Detects and extracts specific facial landmarks (eye corners, nose, mouth corners) for pose estimation.

5. Camera Matrix and Distortion Coefficients
Sets up a camera matrix for mapping 3D points to the 2D image plane. Distortion coefficients are set to zero, assuming no lens distortion.

6. Head Pose Estimation
Calculates the rotation and translation vectors using OpenCV’s solvePnP. Converts the rotation vector to a rotation matrix and decomposes it into Euler angles to get the head's orientation.

7. Direction Determination
Based on the calculated angles:

Negative y angle -> Looking left
Positive y angle -> Looking right
Negative x angle -> Looking down
Positive x angle -> Looking up
No conditions met -> Looking forward

8. Visualization
Displays a line from the nose in the direction the head is oriented and text showing the x, y, and z angles.

9. Display and Exit
Shows the processed frame with detected head pose. Press Esc to exit the loop.

Troubleshooting
Installation errors: Ensure all libraries are correctly installed.
Video capture issues: Check if your webcam is accessible and permissions are enabled.

