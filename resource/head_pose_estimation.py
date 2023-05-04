import os
import cv2
import onnxruntime
import numpy as np
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

class HeadPose(object):

    def __init__(self):
        module_path = os.path.abspath(__file__)
        module_dir = os.path.dirname(module_path)
        model_file_path = os.path.join(module_dir, 'model', 'fsanet.onnx')
        self.fsanet = onnxruntime.InferenceSession(model_file_path, 
            providers=['CPUExecutionProvider'])

        self.expand_factor = 0.6

    def predict(self, input_img, face):

        img_w, img_h = input_img.shape[1], input_img.shape[0]

        (x1, y1, x2, y2) = face
        w = x2 - x1
        h = y2 - y1

        xw1 = max(int(x1 - self.expand_factor * w), 0)
        yw1 = max(int(y1 - self.expand_factor * h), 0)
        xw2 = min(int(x2 + self.expand_factor * w), img_w - 1)
        yw2 = min(int(y2 + self.expand_factor * h), img_h - 1)

        # head image
        data = cv2.resize(input_img[yw1:yw2+1, xw1:xw2+1,:], (64,64))
        # cv2.imshow("head", data)

        data = data.transpose((2,0,1))
        data = np.expand_dims(data,axis=0)
        data = (data-127.5)/128
        data = data.astype(np.float32)
        onnx_input = {self.fsanet.get_inputs()[0].name: data}
        outputs = self.fsanet.run(None, onnx_input)
        yaw, pitch, roll = outputs[0][0]
        return yaw, pitch, roll

if __name__ == '__main__':
    head_pose = HeadPose()