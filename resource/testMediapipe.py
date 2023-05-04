import cv2
import mediapipe as mp
from time import time
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh
# For webcam input:
cap = cv2.VideoCapture("photo/3分钟头部视频.mp4")
run_time = 0
frame_num = cap.get(7)

# with mp_face_detection.FaceDetection(
#     model_selection=0, min_detection_confidence=0.5) as face_detection:
#   while cap.isOpened():
#     success, image = cap.read()
#     if not success:
#       cap.release()
#       cv2.destroyAllWindows()
#       print(f"blazeFace平均运行时间：{run_time / frame_num}")
#       break

#     # To improve performance, optionally mark the image as not writeable to
#     # pass by reference.
#     image.flags.writeable = False
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#     tik = time()
#     results = face_detection.process(image)
#     run_time += time() - tik

#     # Draw the face detection annotations on the image.
#     # image.flags.writeable = True
#     # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
#     # if results.detections:
#     #   for detection in results.detections:
#     #     mp_drawing.draw_detection(image, detection)
#     # # Flip the image horizontally for a selfie-view display.
#     # cv2.imshow('MediaPipe Face Detection', cv2.flip(image, 1))
#     # if cv2.waitKey(5) & 0xFF == 27:
#     #   break
# cap.release()

# For webcam input:
with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      cap.release()
      cv2.destroyAllWindows()
      print(f"face mesh平均运行时间：{run_time / frame_num}")
      break

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    tik = time()
    results = face_mesh.process(image)
    run_time += time() - tik
cap.release()