import cv2
import numpy as np
import mediapipe as mp
from time import time

def cal_eye_ear(self, landmark):
    left_eye_top = landmark[386]
    left_eye_bottom = landmark[374]
    left_eye_left = landmark[263]
    left_eye_right = landmark[362]

    right_eye_top = landmark[159]
    right_eye_bottom = landmark[145]
    right_eye_left = landmark[133]
    right_eye_right = landmark[33]

    mouth_top = landmark[13]
    mouth_bottom = landmark[14]
    mouth_left = landmark[308]
    mouth_right = landmark[78]

    lip_top = landmark[0]
    lip_bottom = landmark[17]
    lip_left = landmark[291]
    lip_right = landmark[61]

    left_eye_ear = self.distance(left_eye_top, left_eye_bottom) / self.distance(left_eye_left, left_eye_right)
    right_eye_ear = self.distance(right_eye_top, right_eye_bottom) / self.distance(right_eye_left, right_eye_right)
    mouth_ear = self.distance(mouth_top, mouth_bottom) / self.distance(mouth_left, mouth_right)
    lip_ear = self.distance(lip_top, lip_bottom) / self.distance(lip_left, lip_right)

    return left_eye_ear, right_eye_ear, mouth_ear, lip_ear

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

# For webcam input:
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
cap = cv2.VideoCapture(0)
img_idx = 0
start_time = time()

with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:
    while cap.isOpened():
        img_idx = img_idx + 1
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image)

        # Draw the face mesh annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:

                h, w, _ = image.shape
                cv2.putText(image, 
                    f'{w} x {h}', 
                    (20,20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
                landmark = face_landmarks.landmark

                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_tesselation_style())
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_contours_style())
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_IRISES,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_iris_connections_style())

        cv2.imshow('MediaPipe Face Mesh', image)
        if cv2.waitKey(1) & 0xFF == 27:
            cv2.destroyAllWindows()
            print(f"fps: {(img_idx / (time() - start_time)):2f}")
            break