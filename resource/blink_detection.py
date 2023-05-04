
import os
import cv2
import dlib

# module_path = os.path.abspath(__file__)
# module_dir = os.path.dirname(module_path)
# model_file_path = os.path.join(module_dir, 'model', 'shape_predictor_68_face_landmarks.dat')

def distance(pointA, pointB):
    return pow((pointA.x - pointB.x) * (pointA.x - pointB.x)\
     + (pointA.y - pointB.y) * (pointA.y - pointB.y), 0.5)

def get_EAR(eye):
    return 0.5 * (distance(eye[1], eye[5]) + distance(eye[2], eye[4]))\
     / distance(eye[0], eye[3])

class EyesDetection(object):
    def __init__(self, detector_type, *model_path):
        self.eye_open_th = 0.15
        self.mouth_open_th = 0.08
        if detector_type == "dlib":
            self.facial_landmark = dlib.shape_predictor(*model_path)
        else:
            raise ValueError("detector_type is NOT a supported value")

    def predict(self, input_img, face):
        img_w = input_img.shape[1]
        img_h = input_img.shape[0]
        ad = 1.6

        face = dlib.rectangle(*face)
        shape = self.facial_landmark(input_img, face)

        right_eye_points = shape.parts()[36:42]
        right_eye_left_point = shape.parts()[36]
        right_eye_right_point = shape.parts()[39]
        right_eye_middle_point = (right_eye_left_point + right_eye_right_point) / 2
        right_eye_radius = (right_eye_right_point.x - right_eye_left_point.x) / 2

        # expand eye area
        r_x1 = max(int(right_eye_middle_point.x - ad * right_eye_radius), 0)
        r_y1 = max(int(right_eye_middle_point.y - ad * right_eye_radius), 0)
        r_x2 = min(int(right_eye_middle_point.x + ad * right_eye_radius), img_w - 1)
        r_y2 = min(int(right_eye_middle_point.y + ad * right_eye_radius), img_h - 1)

        right_eye = (r_x1, r_y1, r_x2, r_y2)

        left_eye_points = shape.parts()[42:48]
        left_eye_left_point = shape.parts()[42]
        left_eye_right_point = shape.parts()[45]
        left_eye_middle_point = (left_eye_left_point + left_eye_right_point) / 2
        left_eye_radius = (left_eye_right_point.x - left_eye_left_point.x) / 2

        l_x1 = max(int(left_eye_middle_point.x - ad * left_eye_radius), 0)
        l_y1 = max(int(left_eye_middle_point.y - ad * left_eye_radius), 0)
        l_x2 = min(int(left_eye_middle_point.x + ad * left_eye_radius), img_w - 1)
        l_y2 = min(int(left_eye_middle_point.y + ad * left_eye_radius), img_h - 1)

        left_eye = (l_x1, l_y1, l_x2, l_y2)

        right_eye_img = input_img[right_eye[1]:right_eye[3], right_eye[0]:right_eye[2], :]
        left_eye_img = input_img[left_eye[1]:left_eye[3], left_eye[0]:left_eye[2], :]
        right_eye_img = cv2.cvtColor(right_eye_img, cv2.COLOR_BGR2GRAY)
        left_eye_img = cv2.cvtColor(left_eye_img, cv2.COLOR_BGR2GRAY)
        right_eye_img = cv2.resize(right_eye_img, (32, 32))
        left_eye_img = cv2.resize(left_eye_img, (32, 32))

        mouth = (shape.parts()[48], shape.parts()[61], shape.parts()[63],
            shape.parts()[54], shape.parts()[65], shape.parts()[67])
        mouth_EAR = get_EAR(mouth)
        mouth_open = True if mouth_EAR > self.mouth_open_th else False

        left_EAR = get_EAR(left_eye_points)
        right_EAR = get_EAR(right_eye_points)
        is_left_eye_closed = False
        is_right_eye_closed = False

        if (left_EAR < self.eye_open_th):
            is_left_eye_closed = True
        if (right_EAR < self.eye_open_th):
            is_right_eye_closed = True

        return right_eye, left_eye, right_eye_img, left_eye_img, is_left_eye_closed, is_right_eye_closed, mouth_open

def draw_eyes(img, left_eye, right_eye, is_left_eye_closed, is_right_eye_closed):

    left_color = (0, 255, 0) if is_left_eye_closed else (255, 0, 0)
    right_color = (0, 255, 0) if is_right_eye_closed else (255, 0, 0)

    cv2.rectangle(img, (right_eye[0], right_eye[1]), (right_eye[2], right_eye[3]), right_color, 3)
    cv2.rectangle(img, (left_eye[0], left_eye[1]), (left_eye[2], left_eye[3]), left_color, 3)

def BlinkDetection(detector_type, *model_path):
    if detector_type == "Lenet5":
        return Lenet5(*model_path)
    elif detector_type == "dlib":
        return DlibDetection()
    else:
        raise ValueError("detector_type is NOT a supported value")

class Lenet5(object):
    def __init__(self, model_path):
        self.net = cv2.dnn.readNetFromONNX(model_path)

    def predict(self, eye_img):

        blob = cv2.dnn.blobFromImage(eye_img, 1.0, (32, 32), (104, 117, 123), swapRB=True, crop=False)
        self.net.setInput(blob)
        result = self.net.forward()[0][0]
        return result

# if __name__ == '__main__':
#     blink_detector = BlinkDetection("model/blink_detection_lenet.onnx")
