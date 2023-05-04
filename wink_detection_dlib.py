#coding=utf-8
#图片检测 - Dlib版本

#dlib对于眼镜效果不好，我决定自己训练一个模型

import cv2
import dlib
import time
import numpy as np
import matplotlib.pyplot as plt
# from scipy import signal

from resource.face_detection import FaceDetection

def distance(pointA, pointB):
    return pow((pointA.x - pointB.x) * (pointA.x - pointB.x)\
     + (pointA.y - pointB.y) * (pointA.y - pointB.y), 0.5)

def get_EAR(eye):
    return 0.5 * (distance(eye[1], eye[5]) + distance(eye[2], eye[4]))\
     / distance(eye[0], eye[3])

#人脸分类器
# detector = dlib.get_frontal_face_detector()
face_proto_path = r"resource\model\deploy.prototxt"
face_model_path = r"resource\model\res10_300x300_ssd_iter_140000.caffemodel"
detector = FaceDetection("ssd", face_proto_path, face_model_path)

# 获取人脸检测器
predictor = dlib.shape_predictor(
    "resource/model/shape_predictor_68_face_landmarks.dat"
)

# capture video
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280*1)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720*1)
start_time=time.time()
left_EAR_list = []
right_EAR_list = []
img_idx = 0

tik = time.time()
part1 = 0
part2 = 0
part3 = 0

while True:
    # get video frame
    ret, input_img = cap.read()
    img_idx = img_idx + 1

    # gray = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
    # dets = detector(gray, 1)
    faces = detector.predict(input_img)
    dets = [dlib.rectangle(*face) for face in faces]

    part1 += time.time() - tik
    tik = time.time()

    for face in dets:
        shape = predictor(input_img, face)

        # part2 += time.time() - tik
        # tik = time.time()

        right_eye = shape.parts()[36:42]
        left_eye = shape.parts()[42:48]
        left_ear = get_EAR(left_eye)
        right_ear = get_EAR(right_eye)
        # eyes = shape.parts()[36:48]
        # left_EAR_list.append(get_EAR(left_eye))
        # right_EAR_list.append(get_EAR(right_eye))
        mouth = (shape.parts()[48], shape.parts()[61], shape.parts()[63],
            shape.parts()[54], shape.parts()[65], shape.parts()[67])
        mouth_EAR = get_EAR(mouth)
        # if (mouth_EAR > 0.2):
        #     mouth_open = True
        # tag = 'closed' if trainning_set_y[frame_pos + 1] else 'opened'


        cv2.putText(input_img, 
            f'{mouth_EAR:.2f}', 
            (shape.parts()[57].x, shape.parts()[57].y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255))

        cv2.putText(input_img, 
            f'{left_ear:.2f}', 
            (shape.parts()[36].x, shape.parts()[36].y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255))

        cv2.putText(input_img, 
            f'{right_ear:.2f}', 
            (shape.parts()[42].x, shape.parts()[42].y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255))

        for pt in shape.parts():
            pt_pos = (pt.x, pt.y)
            cv2.circle(input_img, pt_pos, 1, (0, 255, 0), 2)
        cv2.imshow("image", input_img)

        # part3 += time.time() - tik
        # tik = time.time()

    if cv2.waitKey(1) == ord('q'): # 按'q'键推出
        
        # total_time = time.time() - start_time
        # print(f"FPS: {1 / (total_time / img_idx)}, {part1}, {part2}, {part3}")
        # plt.plot(left_EAR_list)
        # plt.plot(right_EAR_list)
        # plt.show()
        break

# path = "resource/photo/me.jpg"
# img = cv2.imread(path)
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# #人脸分类器
# detector = dlib.get_frontal_face_detector()
# # 获取人脸检测器
# predictor = dlib.shape_predictor(
#     "resource/model/shape_predictor_68_face_landmarks.dat"
# )

# dets = detector(gray, 1)
# for face in dets:
#     shape = predictor(img, face)  # 寻找人脸的68个标定点
#     # 遍历所有点，打印出其坐标，并圈出来
#     right_eye = shape.parts()[36:42]
#     left_eye = shape.parts()[42:48]
#     eyes = shape.parts()[36:48]
#     left_EAR = get_EAR(left_eye)
#     right_EAR = get_EAR(right_eye)
#     for pt in eyes:
#         pt_pos = (pt.x, pt.y)
#         cv2.circle(img, pt_pos, 1, (0, 255, 0), 2)
#     cv2.imshow("image", img)
# print('所用时间为{}'.format(time.time()-t))
# print(f'左眼宽高比：{left_EAR}，右眼宽高比：{right_EAR}')
# cv2.waitKey(0)
# # time.sleep(5)
# cv2.destroyAllWindows()