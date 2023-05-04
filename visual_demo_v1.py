# visual mouse demo v1

# face detection: ssd_face 
# size: 10 MB
# input: 300x300x3
# from: Caffe model, opencv inference

# blink detection: Lenet5
# size 182 KB
# input: 32x32x1
# from: keras model, convert to onnx, opencv inference

# head pose estimation: fsanet
# size: 1.2MB
# input: 64x64x3
# from: keras model, convert to onnx, onnxruntime inference

import cv2

from threading import Thread

from resource.face_detection import FaceDetection
from resource.blink_detection import BlinkDetection, EyesDetection, draw_eyes
from resource.head_pose_estimation import HeadPose, draw_axis

from time import time

class VisualModule(object):
    """docstring for VisualModule"""
    def __init__(self, message_queue):
        self.message_queue = message_queue

        face_proto_path = r"resource\model\deploy.prototxt"
        face_model_path = r"resource\model\res10_300x300_ssd_iter_140000.caffemodel"
        self.face_detector = FaceDetection("ssd", face_proto_path, face_model_path)

        eyes_model_path = r"resource\model\shape_predictor_68_face_landmarks.dat"
        self.eyes_detector = EyesDetection("dlib", eyes_model_path)

        blink_model_path = r"resource\model\blink_detection_lenet-1.6.onnx"
        self.blink_detector = BlinkDetection("Lenet5", blink_model_path)

        self.head_pose = HeadPose()

        self.is_EXIT = False
        Thread(target=self.head_info_detect_loop).start()

    def EXIT(self):
        self.is_EXIT = True

    def head_info_detect_loop(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280*1)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720*1)

        while not self.is_EXIT:
            ret, frame = cap.read()
            faces = self.face_detector.predict(frame)
            if len(faces) < 1:
                self.message_queue.put({})
            else:
                for face in faces:
                    right_eye, left_eye, right_eye_img, \
                    left_eye_img, is_left_eye_closed, is_right_eye_closed, \
                    mouth_open = self.eyes_detector.predict(frame, face)
                    if mouth_open:
                        is_left_eye_closed = self.blink_detector.predict(left_eye_img)
                        is_right_eye_closed = self.blink_detector.predict(right_eye_img)

                        yaw, pitch, roll = self.head_pose.predict(frame, face)

                        middle_face_x = (face[0] + face[2]) / 2
                        middle_face_y = (face[1] + face[3]) / 2
                        draw_eyes(frame, left_eye, right_eye, is_left_eye_closed, is_right_eye_closed)
                        draw_axis(frame, yaw, pitch, roll, tdx=middle_face_x, tdy=middle_face_y)

                        self.message_queue.put({
                                "head_pose": (yaw, pitch, roll),
                                "is_left_eye_closed": is_left_eye_closed,
                                "is_right_eye_closed": is_right_eye_closed,
                            })
                        break
                    else:
                        self.message_queue.put({})

            frame = cv2.resize(frame, (640, 360))
            cv2.imshow("visual demo v1", frame)
            key = cv2.waitKey(1)
            # if key == "q":
            #     self.is_EXIT = True
        cv2.destroyAllWindows()


if __name__ == '__main__':

    face_proto_path = r"resource\model\deploy.prototxt"
    face_model_path = r"resource\model\res10_300x300_ssd_iter_140000.caffemodel"
    face_detector = FaceDetection("ssd", face_proto_path, face_model_path)

    eyes_model_path = r"resource\model\shape_predictor_68_face_landmarks.dat"
    eyes_detector = EyesDetection("dlib", eyes_model_path)

    blink_model_path = r"resource\model\blink_detection_lenet-1.6.onnx"
    blink_detector = BlinkDetection("Lenet5", blink_model_path)

    head_pose = HeadPose()

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280*1)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720*1)
    img_idx = 0
    start_time = time()

    while True:
        img_idx = img_idx + 1
        ret, frame = cap.read()

        faces = face_detector.predict(frame)

        for face in faces:
            right_eye, left_eye, right_eye_img, \
            left_eye_img, is_left_eye_closed, is_right_eye_closed, \
            mouth_open = eyes_detector.predict(frame, face)
            # is_left_eye_closed = blink_detector.predict(left_eye_img)
            # is_right_eye_closed = blink_detector.predict(right_eye_img)

            yaw, pitch, roll = head_pose.predict(frame, face)

            middle_face_x = (face[0] + face[2]) / 2
            middle_face_y = (face[1] + face[3]) / 2
            draw_eyes(frame, left_eye, right_eye, is_left_eye_closed, is_right_eye_closed)
            draw_axis(frame, yaw, pitch, roll, tdx=middle_face_x, tdy=middle_face_y)
            break
        cv2.imshow("visual demo v1", frame)
        if cv2.waitKey(1) == ord('q'): # 按'q'键推出
            total_time = time() - start_time
            print(f"FPS: {img_idx / total_time}")
            break
    cv2.destroyAllWindows()



