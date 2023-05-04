# visual mouse demo v2
# mediapipe version

import cv2
import numpy as np
import mediapipe as mp

from time import time
from threading import Thread

from resource.head_pose_estimation import draw_axis
from resource.smooth import Smooth

class VisualModule(object):
    def __init__(self, message_queue):

        self.w = 640
        self.h = 480
        self.smoother = Smooth(['lip_direction_now', 'yaw', 'pitch', 'roll', 'lip_ear'], 5, smooth_type="pow")
        self.message_queue = message_queue
        # self.head_pose = HeadPose()
        self.is_EXIT = False

        Thread(target=self.head_info_detect_loop).start()

        # left_eye_top = 386
        # left_eye_bottom = 374
        # left_eye_left = 263
        # left_eye_right = 362

        # right_eye_top = 159
        # right_eye_bottom = 145
        # right_eye_left = 133
        # right_eye_right = 33
        # self.left = frozenset([(left_eye_top, left_eye_left),(left_eye_left, left_eye_bottom),
        #     (left_eye_bottom, left_eye_right),(left_eye_right, left_eye_top)])
        # self.right = frozenset([(right_eye_top, right_eye_left),(right_eye_left, right_eye_bottom),
        #     (right_eye_bottom, right_eye_right),(right_eye_right, right_eye_top)])

    def EXIT(self):
        self.is_EXIT = True

    def distance(self, A, B):
        return pow((A.x-B.x)*(A.x-B.x) + (A.y-B.y)*(A.y-B.y) + (A.z-B.z)*(A.z-B.z), 0.5)

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

    def lip_direction(self, landmark):
        # origin coordinate is not system of rectangular coordinate, thus we need to convert the corrdinate
        points = self.convert_to_cartesian(landmark)

        # middle_lip = (points[0] + points[11] + points[12] + points[13] + 
        #     points[14] + points[15] + points[16] + points[17]) / 8

        forehead = points[10]
        chin = points[152]
        nose_tip = points[1]

        # solve bisector of head
        # vec_nose_to_chin = chin - nose_tip
        # vec_nose_to_forehead = forehead - nose_tip
        # normal = np.cross(vec_nose_to_chin, vec_nose_to_forehead)
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
        cap = cv2.VideoCapture(0)
        img_idx = 0
        start_time = time()

        with mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as face_mesh:
            while cap.isOpened() and not self.is_EXIT:
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

                        h, w, _ = image.shape
                        cv2.putText(image, 
                            f'{w} x {h}', 
                            (20,20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
                        landmark = face_landmarks.landmark

                        # x1 = int(min([point.x for point in landmark]) * w)
                        # y1 = int(min([point.y for point in landmark]) * h)
                        # x2 = int(max([point.x for point in landmark]) * w)
                        # y2 = int(max([point.y for point in landmark]) * h)

                        # yaw, pitch, roll = self.head_pose.predict(image, (x1,y1,x2,y2))
                        # cv2.putText(image, 
                        #     f'{yaw:.2f}, {pitch:.2f}, {roll:.2f}', 
                        #     (20,80), 
                        #     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
                        # right_cheek = np.array([landmark[234].x, landmark[234].y * 0.75, landmark[234].z])
                        # left_cheek = np.array([landmark[454].x, landmark[454].y * 0.75, landmark[454].z])
                        roll = np.rad2deg(np.arctan((landmark[234].y-landmark[454].y)*0.75/(landmark[234].x-landmark[454].x)))
                        yaw = -np.rad2deg(np.arctan((landmark[234].z-landmark[454].z)/(landmark[234].x-landmark[454].x)))
                        pitch = -np.rad2deg(np.arctan((landmark[10].z-landmark[152].z)/((landmark[10].y-landmark[152].y)*0.75)))

                        cv2.putText(image, 
                            f'{yaw:.2f}, {pitch:.2f}, {roll:.2f}', 
                            (20,110), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
                        # middle_face_x = (x1 + x2) / 2
                        # middle_face_y = (y1 + y2) / 2

                        left_pos = (int(landmark[374].x * w), int(landmark[374].y * h))
                        right_pos = (int(landmark[145].x * w), int(landmark[145].y * h))
                        mouth_pos = (int(landmark[14].x * w), int(landmark[14].y * h))
                        lip_pos = (int(landmark[17].x * w), int(landmark[17].y * h))
                        l_ear, r_ear, m_ear, lip_ear = self.cal_eye_ear(landmark)

                        is_left_eye_closed = True if l_ear<0.18 else False
                        is_right_eye_closed = True if r_ear<0.18 else False
                        is_mouth_closed = True if m_ear<0.05 else False
                        is_lip_closed = True if lip_ear<0.3 else False

                        lip_direction_now, forehead, chin, nose_tip = self.lip_direction(landmark)
                        # out_dict = self.smoother.smooth(lip_direction_now=lip_direction_now, yaw=yaw, pitch=pitch, roll=roll, lip_ear=lip_ear)
                        # lip_direction_now = out_dict['lip_direction_now']

                        # cv2.line(image, forehead[:2].astype("int32"), chin[:2].astype("int32"), (255,0,0))
                        # cv2.line(image, forehead[:2].astype("int32"), nose_tip[:2].astype("int32"), (255,0,0))
                        # cv2.line(image, nose_tip[:2].astype("int32"), chin[:2].astype("int32"), (255,0,0))

                        # self.depth_detection(landmark, image)
                        # self.z_detection(landmark, image)

                        draw_axis(image, yaw, pitch, roll, tdx=int(landmark[5].x * w), tdy=int(landmark[5].y * h))
                        # draw_axis(image, out_dict['yaw'], out_dict['pitch'], out_dict['roll'], tdx=int(landmark[5].x * w), tdy=int(landmark[5].y * h))
                        # cv2.rectangle(image, (x1, y1), (x2, y2), (0,255,0), 3)

                        # cv2.putText(image, 
                        #     f'{l_ear:.2f}', 
                        #     left_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
                        # cv2.putText(image, 
                        #     f'{r_ear:.2f}', 
                        #     right_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
                        # cv2.putText(image, 
                        #     f'{m_ear:.2f}', 
                        #     mouth_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
                        # cv2.putText(image, 
                        #     f'{lip_ear:.2f}', 
                        #     mouth_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
                        # cv2.putText(image, 
                        #     # 'left' if lip_direction_now>0 else 'right', 
                        #     f"{lip_direction_now:.2f}",
                        #     (int(landmark[200].x * w), int(landmark[200].y * h)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))

                        # cv2.putText(image, 
                        #     f'{int(landmark[374].z * w)}', 
                        #     (int(landmark[374].x * w), int(landmark[374].y * h)), 
                        #     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
                        # cv2.putText(image, 
                        #     f'{int(landmark[0].z * w)}', 
                        #     (int(landmark[0].x * w), int(landmark[0].y * h)), 
                        #     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
                        # cv2.putText(image, 
                        #     f'{int(landmark[145].z * w)}', 
                        #     (int(landmark[145].x * w), int(landmark[145].y * h)), 
                        #     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
                        # cv2.putText(image, 
                        #     f'{int(landmark[1].z * w)}', 
                        #     (int(landmark[1].x * w), int(landmark[1].y * h)), 
                        #     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))

                        # horizontal line of face
                        # cv2.putText(image, 
                        #     f'{int(landmark[234].z * w)}', 
                        #     (int(landmark[234].x * w), int(landmark[234].y * h)), 
                        #     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
                        # cv2.putText(image, 
                        #     f'{int(landmark[454].z * w)}', 
                        #     (int(landmark[454].x * w), int(landmark[454].y * h)), 
                        #     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))

                        # vertical line of face
                        # cv2.putText(image, 
                        #     f'{int(landmark[10].z * w)}', 
                        #     (int(landmark[10].x * w), int(landmark[10].y * h)), 
                        #     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
                        # cv2.putText(image, 
                        #     f'{int(landmark[152].z * w)}', 
                        #     (int(landmark[152].x * w), int(landmark[152].y * h)), 
                        #     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))

                        self.message_queue.put({
                                "head_pose": (yaw, pitch, roll),
                                "is_left_eye_closed": is_left_eye_closed,
                                "is_right_eye_closed": is_right_eye_closed,
                                "is_mouth_closed": is_mouth_closed,
                                "is_lip_closed": is_lip_closed,
                                "lip_direction": lip_direction_now,
                            })

                        # mp_drawing.draw_landmarks(
                        #     image=image,
                        #     landmark_list=face_landmarks,
                        #     connections=self.left,
                        #     landmark_drawing_spec=None)
                        # mp_drawing.draw_landmarks(
                        #     image=image,
                        #     landmark_list=face_landmarks,
                        #     connections=self.right,
                        #     landmark_drawing_spec=None)

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
                # cv2.imshow('MediaPipe Face Mesh', cv2.flip(image, 1))
                
                else:
                    self.message_queue.put({})

                cv2.imshow('MediaPipe Face Mesh', cv2.resize(image, (400,300)))
                if cv2.waitKey(1) & 0xFF == 27:
                    cv2.destroyAllWindows()
                    print(f"fps: {(img_idx / (time() - start_time)):2f}")
                    self.is_EXIT = True
        cap.release()

if __name__ == '__main__':
    class placeholder(object):
        """docstring for placeholder"""
        def put(self, o):
            pass
            
    visual_module = VisualModule(placeholder())
    visual_module.head_info_detect_loop()