import mediapipe as mp
import cv2
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

# Define the 3D model points of the landmarks
model_points = np.array([
    (0.0, 0.0, 0.0),           # Nose tip
    (0.0, -330.0, -65.0),      # Chin
    (-225.0, 170.0, -135.0),   # Left eye left corner
    (225.0, 170.0, -135.0),    # Right eye right corner
    (-150.0, -150.0, -125.0),  # Left Mouth corner
    (150.0, -150.0, -125.0)    # Right mouth corner
])

# Define the camera intrinsic parameters
camera_matrix = np.array([
    [640.0, 0.0, 320.0],
    [0.0, 640.0, 240.0],
    [0.0, 0.0, 1.0]
])

# Define the distortion coefficients
dist_coeffs = np.zeros((4,1))

# Initialize the Face Mesh model
face_mesh = mp_face_mesh.FaceMesh()

# Initialize the video capture
cap = cv2.VideoCapture(0)

while cap.isOpened():
    # Read a frame from the video capture
    success, image = cap.read()
    if not success:
        break

    # Flip the image horizontally for a mirror effect
    image = cv2.flip(image, 1)

    # Convert the image to RGB format
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Run the Face Mesh model to detect landmarks
    results = face_mesh.process(image_rgb)
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Convert the landmarks to a numpy array
            landmarks = np.array([(lm.x, lm.y, lm.z) for lm in face_landmarks.landmark])

            # Estimate the head pose using solvePnP
            success, rotation_vector, translation_vector = cv2.solvePnP(model_points, landmarks, camera_matrix, dist_coeffs)

            # Draw the pose axes on the image
            axis_length = 100
            nose_tip = landmarks[0]
            imgpts, jac = cv2.projectPoints(axis_length * np.array([[0, 0, 0], [0, 0, -axis_length], [0, axis_length, 0]]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)
            img = cv2.line(image, tuple(nose_tip[:2].astype(int)), tuple(imgpts[1].ravel().astype(int)), (0,0,255), 3)
            img = cv2.line(img, tuple(nose_tip[:2].astype(int)), tuple(imgpts[2].ravel().astype(int)), (0,255,0), 3)

    # Display the image
    cv2.imshow('Face Mesh', image)
    if cv2.waitKey(1) & 0xFF == 27:
        cap.release()
        cv2.destroyAllWindows()
        break