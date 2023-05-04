import os
import cv2
import numpy as np
import mediapipe as mp

def FaceDetection(detector_type, *model_path):
    if detector_type == "ssd":
        return SSD_detector(*model_path)
    if detector_type == "mediapipe":
        return MediaPipe_detector()
    else:
        raise ValueError("detector_type is NOT a supported value")

def expand_face_to_head(face, img_w, img_h):
    ad = 0.6

    w = face[1][0] - face[0][0]
    h = face[1][1] - face[0][1]

    xw1 = max(int(face[0][0] - ad * w), 0)
    yw1 = max(int(face[0][1] - ad * h), 0)
    xw2 = min(int(face[1][0] + ad * w), img_w - 1)
    yw2 = min(int(yface[1][1] + ad * h), img_h - 1)

def draw_faces(faces, img):
    for face in faces:
        cv2.rectangle(img, (face[0], face[1]), (face[2], face[3]), (0, 255, 0), 3)
    cv2.imshow("show face", img)
    cv2.waitKey()
    cv2.destroyAllWindows()

class SSD_detector(object):
    def __init__(self, proto_path, model_path):
        # ssd model defined with Caffe
        self.detector = cv2.dnn.readNetFromCaffe(proto_path, model_path)

    # opencv
    def predict(self, input_img):
        blob = cv2.dnn.blobFromImage(cv2.resize(input_img, (300, 300)), 1.0,
                (300, 300), (104.0, 177.0, 123.0))
        self.detector.setInput(blob)
        result = self.detector.forward()

        faces = []

        # loop over the detections
        if result.shape[2]>0:
            for i in range(0, result.shape[2]):
                # extract the confidence (i.e., probability) associated with the
                # prediction
                confidence = result[0, 0, i, 2]

                # filter out weak detections
                if confidence > 0.5:
                    # compute the (x, y)-coordinates of the bounding box for
                    # the face and extract the face ROI
                    (h0, w0) = input_img.shape[:2]
                    box = result[0, 0, i, 3:7] * np.array([w0, h0, w0, h0])
                    # (x1, y1, x2, y2) = box.astype("int")

                    faces.append(box.astype("int"))

        # find the biggest face in image
        faces = sorted(faces, 
            key=lambda face:(face[2]-face[0])*(face[3]-face[1]), 
            reverse=True)
        return faces

if __name__ == '__main__':
    # detector = FaceDetection("ssd", "model/deploy.prototxt", 
    #     "model/res10_300x300_ssd_iter_140000.caffemodel")
    # me = cv2.imread("photo/me.jpg")
    # faces = detector.detect(me)
    # draw_faces(faces, me)

    from time import time

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

            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(image)

            # Draw the face mesh annotations on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_LEFT_EYE,
                        landmark_drawing_spec=None)
                    mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_RIGHT_EYE,
                        landmark_drawing_spec=None)
                    # mp_drawing.draw_landmarks(
                    #     image=image,
                    #     landmark_list=face_landmarks,
                    #     connections=mp_face_mesh.FACEMESH_CONTOURS,
                    #     landmark_drawing_spec=None,
                    #     connection_drawing_spec=mp_drawing_styles
                    #     .get_default_face_mesh_contours_style())
                    # mp_drawing.draw_landmarks(
                    #     image=image,
                    #     landmark_list=face_landmarks,
                    #     connections=mp_face_mesh.FACEMESH_IRISES,
                    #     landmark_drawing_spec=None,
                    #     connection_drawing_spec=mp_drawing_styles
                    #     .get_default_face_mesh_iris_connections_style())
                    # print(face_landmarks.landmark[468], face_landmarks.landmark[473])
                    # left eye {384, 385, *386*, 387, 388, 390, *263*, *362*, 398, 466, 373, *374*, 249, 380, 381, 382}
                    # right eye {160, *33*, 161, 163, *133*, 7, 173, 144, *145*, 246, 153, 154, 155, 157, 158, *159*}
            # Flip the image horizontally for a selfie-view display.
            cv2.imshow('MediaPipe Face Mesh', cv2.flip(image, 1))
            if cv2.waitKey(1) & 0xFF == 27:
                cv2.destroyAllWindows()
                print(f"fps: {(img_idx / (time() - start_time)):2f}")
                break
    cap.release()