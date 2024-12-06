# visual mouse demo v2
# mediapipe version

import cv2
# import win32gui, win32con
import numpy as np
import mediapipe as mp

from threading import Thread
from utils.smooth import Smooth
from utils.event_system import event_system
from utils.config_utils import config_utils

from math import cos, sin

def draw_axis(img, yaw, pitch, roll, tdx=None, tdy=None, size = 50):
    # print(yaw,roll,pitch)
    pitch = pitch * np.pi / 180
    yaw = -(yaw * np.pi / 180)
    roll = roll * np.pi / 180

    if tdx != None and tdy != None:
        tdx = tdx
        tdy = tdy
    else:
        height, width = img.shape[:2]
        tdx = width / 2
        tdy = height / 2

    # X-Axis pointing to right. drawn in red
    x1 = size * (cos(yaw) * cos(roll)) + tdx
    y1 = size * (cos(pitch) * sin(roll) + cos(roll) * sin(pitch) * sin(yaw)) + tdy

    # Y-Axis | drawn in green
    #        v
    x2 = size * (-cos(yaw) * sin(roll)) + tdx
    y2 = size * (cos(pitch) * cos(roll) - sin(pitch) * sin(yaw) * sin(roll)) + tdy

    # Z-Axis (out of the screen) drawn in blue
    x3 = size * (sin(yaw)) + tdx
    y3 = size * (-cos(yaw) * sin(pitch)) + tdy

    cv2.line(img, (int(tdx), int(tdy)), (int(x1),int(y1)),(0,0,255),3)
    cv2.line(img, (int(tdx), int(tdy)), (int(x2),int(y2)),(0,255,0),3)
    cv2.line(img, (int(tdx), int(tdy)), (int(x3),int(y3)),(255,0,0),2)

    return img

def distance(A, B):
        return pow((A.x-B.x)*(A.x-B.x) + (A.y-B.y)*(A.y-B.y) + (A.z-B.z)*(A.z-B.z), 0.5)

class VisualModule(object):
    def __init__(self):
        self.smoother = Smooth(['yaw', 'pitch', 'roll'], 1, smooth_type="mean")
        self.init_arg()
        event_system.subscribe("restart", self.init_arg)

        Thread(target=self.head_info_detect_loop).start()

    def init_arg(self):
        self.is_EXIT = False

        visual_arg = config_utils["control_arg"]

        self.w                      = eval(visual_arg["resolution_w"])
        self.h                      = eval(visual_arg["resolution_h"])
        self.is_flip                = eval(visual_arg["is_flip"])
        # self.is_draw_aspect_ratio   = eval(visual_arg["is_draw_aspect_ratio"])
        self.is_draw_aspect_ratio   = True
        self.camera_index           = eval(visual_arg["camera_index"])

    def EXIT(self):
        print(f'visual module stop')
        self.is_EXIT = True

    # aspect ratio
    def cal_aspect_ratio(self, landmark):
        right_eye_top = landmark[386]
        right_eye_bottom = landmark[374]
        right_eye_left = landmark[263]
        right_eye_right = landmark[362]

        left_eye_top = landmark[159]
        left_eye_bottom = landmark[145]
        left_eye_left = landmark[133]
        left_eye_right = landmark[33]

        mouth_top = landmark[13]
        mouth_bottom = landmark[14]
        mouth_left = landmark[308]
        mouth_right = landmark[78]

        lip_top = landmark[0]
        lip_bottom = landmark[17]
        lip_left = landmark[291]
        lip_right = landmark[61]

        left_eye_aspect_ratio = distance(left_eye_top, left_eye_bottom) / distance(left_eye_left, left_eye_right)
        right_eye_aspect_ratio = distance(right_eye_top, right_eye_bottom) / distance(right_eye_left, right_eye_right)
        mouth_EAR = distance(mouth_top, mouth_bottom) / distance(mouth_left, mouth_right)
        lip_aspect_ratio = distance(lip_top, lip_bottom) / distance(lip_left, lip_right)

        return left_eye_aspect_ratio, right_eye_aspect_ratio, mouth_EAR, lip_aspect_ratio

    # 返回嘴唇底端离面部中轴线的距离
    def lip_direction(self, landmark):
        # origin coordinate is not system of rectangular coordinate, thus we need to convert the corrdinate
        points = self.convert_to_cartesian(landmark)

        forehead = points[10]
        chin = points[152]
        nose_tip = points[1]

        normal = points[454] - points[234]
        magnitude = np.linalg.norm(normal)
        normal = normal / magnitude

        # point-normal type plane function, if result greater then 0, point(x,y,z) is in left of face
        plane = lambda point : normal[0]*(point[0]-forehead[0]) + normal[1]*(point[1]-forehead[1]) + normal[2]*(point[2]-forehead[2])

        return plane(points[17]), forehead, chin, nose_tip

    def convert_to_cartesian(self, landmark):
        cos_yaw = abs((landmark[234].z-landmark[454].z)/(landmark[234].x-landmark[454].x))
        cos_pitch = abs((landmark[10].z-landmark[152].z)/((landmark[10].y-landmark[152].y)*0.75))

        points = np.array([(p.x, p.y, p.z) for p in landmark])
        # points[:,2] = points[:,2] * cos_yaw * cos_pitch
        points[:,2] = points[:,2] * self.w
        points[:,0] = points[:,0] * self.w
        points[:,1] = points[:,1] * self.h
        return points

    def depth_detection(self, landmark, image):
        cos_yaw = abs((landmark[234].z-landmark[454].z)/(landmark[234].x-landmark[454].x))
        cos_pitch = abs((landmark[10].z-landmark[152].z)/((landmark[10].y-landmark[152].y)*0.75))

        points = np.array([(p.x, p.y, p.z) for p in landmark])
        points[:,2] = points[:,2] * cos_yaw * cos_pitch
        points[:,2] = 255 * (points[:,2] - np.min(points[:,2])) / (np.max(points[:,2] - np.min(points[:,2])))
        points[:,0] = points[:,0] * self.w
        points[:,1] = points[:,1] * self.h
        points = points.astype("int32")
        for point in points:
            cv2.circle(image, point[:2], 2, (int(point[2]),int(point[2]),int(point[2])), 20)

    def z_detection(self, landmark, image):
        points = np.array([(p.x, p.y, p.z) for p in landmark])
        points[:,2] = 255 * (points[:,2] - np.min(points[:,2])) / (np.max(points[:,2] - np.min(points[:,2])))
        points[:,0] = points[:,0] * self.w
        points[:,1] = points[:,1] * self.h
        points = points.astype("int32")
        for point in points:
            cv2.circle(image, point[:2], 2, (int(point[2]),int(point[2]),int(point[2])), 20)

    def head_info_detect_loop(self):
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles
        mp_face_mesh = mp.solutions.face_mesh

        # For webcam input:
        drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
        cap = cv2.VideoCapture(self.camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.h)

        # img_idx = 0
        # start_time = time()

        window_name = "Visual Mouse"
        # cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        # cv2.resizeWindow(window_name, self.w, self.h)

        with mp_face_mesh.FaceMesh(
            refine_landmarks=True) as face_mesh:
            while cap.isOpened() and not self.is_EXIT:
                # img_idx = img_idx + 1
                success, image = cap.read()
                if not success:
                    print("请检查摄像头是否安装正确")
                    continue

                if self.is_flip:
                    image = cv2.flip(image, 1)
                    
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

                        h, w, _ = image.shape
                        cv2.putText(image, 
                            f'{w} x {h}', 
                            (20,20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
                        landmark = face_landmarks.landmark

                        roll = np.rad2deg(np.arctan((landmark[234].y-landmark[454].y)*0.75/(landmark[234].x-landmark[454].x)))
                        yaw = -np.rad2deg(np.arctan((landmark[234].z-landmark[454].z)/(landmark[234].x-landmark[454].x)))
                        pitch = -np.rad2deg(np.arctan((landmark[10].z-landmark[152].z)/((landmark[10].y-landmark[152].y)*0.75)))

                        cv2.putText(image, 
                            f'{yaw:.2f}, {pitch:.2f}, {roll:.2f}', 
                            (20,110), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))

                        left_pos = (int(landmark[374].x * w), int(landmark[374].y * h))
                        right_pos = (int(landmark[145].x * w), int(landmark[145].y * h))
                        mouth_pos = (int(landmark[14].x * w), int(landmark[14].y * h))
                        lip_pos = (int(landmark[17].x * w), int(landmark[17].y * h))
                        l_EAR, r_EAR, mouth_AR, lip_AR = self.cal_aspect_ratio(landmark)

                        is_right_eye_closed = True if l_EAR<0.18 else False
                        is_left_eye_closed = True if r_EAR<0.18 else False
                        is_mouth_closed = True if mouth_AR<0.05 else False
                        is_lip_closed = True if lip_AR<0.3 else False

                        lip_direction_now, forehead, chin, nose_tip = self.lip_direction(landmark)

                        out_dict = self.smoother.smooth(yaw=yaw, pitch=pitch, roll=roll)
                        # self.depth_detection(landmark, image)
                        # self.z_detection(landmark, image)

                        yaw = out_dict['yaw']
                        pitch = out_dict['pitch']
                        roll = out_dict['roll']

                        draw_axis(image, yaw, pitch, roll, tdx=int(landmark[5].x * w), tdy=int(landmark[5].y * h))

                        # draw aspect ratio
                        if self.is_draw_aspect_ratio:
                            cv2.putText(image, 
                                f'{l_EAR:.2f}', 
                                left_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
                            cv2.putText(image, 
                                f'{r_EAR:.2f}', 
                                right_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
                            cv2.putText(image, 
                                f'{mouth_AR:.2f}', 
                                mouth_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
                            cv2.putText(image, 
                                f'{lip_AR:.2f}', 
                                lip_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
                            cv2.putText(image, 
                                'left' if lip_direction_now>0 else 'right', 
                                # f"{lip_direction_now:.2f}",
                                (int(landmark[200].x * w), int(landmark[200].y * h)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))

                        event_system.publish("control_info", {
                            "head_pose": (yaw, pitch, roll),
                            "l_EAR": l_EAR,
                            "r_EAR": r_EAR,
                            "mouth_AR": mouth_AR,
                            "lip_AR": lip_AR,

                            "is_left_eye_closed": is_left_eye_closed,
                            "is_right_eye_closed": is_right_eye_closed,
                            "is_mouth_closed": is_mouth_closed,
                            "is_lip_closed": is_lip_closed,
                            "lip_direction": lip_direction_now,
                        })

                        # mp_drawing.draw_landmarks(
                        #     image=image,
                        #     landmark_list=face_landmarks,
                        #     connections=mp_face_mesh.FACEMESH_TESSELATION,
                        #     landmark_drawing_spec=None,
                        #     connection_drawing_spec=mp_drawing_styles
                        #     .get_default_face_mesh_tesselation_style())

                        mp_drawing.draw_landmarks(
                            image=image,
                            landmark_list=face_landmarks,
                            connections=mp_face_mesh.FACEMESH_CONTOURS,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=mp_drawing_styles
                            .get_default_face_mesh_contours_style())

                        # print(face_landmarks.landmark[468], face_landmarks.landmark[473])
                        # left eye {384, 385, *386*, 387, 388, 390, *263*, *362*, 398, 466, 373, *374*, 249, 380, 381, 382}
                        # right eye {160, *33*, 161, 163, *133*, 7, 173, 144, *145*, 246, 153, 154, 155, 157, 158, *159*}
                
                else:
                    event_system.publish("control_info", {})

                event_system.publish("camera_preview", image)

                # cv2.imshow(window_name, cv2.resize(image, (self.w,self.h)))
                # hwnd = win32gui.FindWindow(None, window_name)
                # win32gui.SetWindowPos(hwnd, win32con.HWND_TOPMOST, 0, 0, 0, 0,
                #           win32con.SWP_NOMOVE | win32con.SWP_NOSIZE)

                # key = cv2.waitKey(1)
                # if key & 0xFF == 27:
                #     cv2.destroyAllWindows()
                #     # print(f"fps: {(img_idx / (time() - start_time)):2f}")
                #     event_system.publish('exit', True)
        cap.release()

    def change_camera(self, target_camera):
        self.camera_index = self.camera_index + 1
        pass

if __name__ == '__main__':
    import configparser
    config = configparser.ConfigParser()
    config.read("conf.ini")
    visual_module = VisualModule(config)