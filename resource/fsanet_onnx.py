import onnxruntime
import numpy as np
import cv2
from time import time
from face_detection import FaceDetection

def draw_axis(img, yaw, pitch, roll, tdx=None, tdy=None, size = 50,thickness=(2,2,2)):
    """
    Function used to draw y (headpose label) on Input Image x.
    Implemented by: shamangary
    https://github.com/shamangary/FSA-Net/blob/master/demo/demo_FSANET.py
    Modified by: Omar Hassan
    """
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
    x1 = size * (np.cos(yaw) * np.cos(roll)) + tdx
    y1 = size * (np.cos(pitch) * np.sin(roll) + np.cos(roll) * np.sin(pitch) * np.sin(yaw)) + tdy

    # Y-Axis | drawn in green
    #        v
    x2 = size * (-np.cos(yaw) * np.sin(roll)) + tdx
    y2 = size * (np.cos(pitch) * np.cos(roll) - np.sin(pitch) * np.sin(yaw) * np.sin(roll)) + tdy

    # Z-Axis (out of the screen) drawn in blue
    x3 = size * (np.sin(yaw)) + tdx
    y3 = size * (-np.cos(yaw) * np.sin(pitch)) + tdy

    cv2.line(img, (int(tdx), int(tdy)), (int(x1),int(y1)),(0,0,255),thickness[0])
    cv2.line(img, (int(tdx), int(tdy)), (int(x2),int(y2)),(0,255,0),thickness[1])
    cv2.line(img, (int(tdx), int(tdy)), (int(x3),int(y3)),(255,0,0),thickness[2])

    return img

def FaceDetector():
    proto_path = r"model\deploy.prototxt"
    model_path = r"model\res10_300x300_ssd_iter_140000.caffemodel"
    return FaceDetection("ssd", proto_path, model_path)
    
# data = cv2.dnn.blobFromImage(cv2.resize(input_img, (64, 64)), 1.0, (64, 64), (104.0, 177.0, 123.0))
def one_image():
    ad = 0.6

    detector = FaceDetector()
    fsanet = onnxruntime.InferenceSession('model/fsanet.onnx', providers=['CPUExecutionProvider'])
    # ssd = onnxruntime.InferenceSession('face_detection_ssd.onnx', providers=['CPUExecutionProvider'])
    input_img = cv2.imread('photo/test.jpg')
    img_w, img_h = input_img.shape[1], input_img[0]

    faces = detector.predict(input_img)

    (x1, y1, x2, y2) = faces[0]
    w = x2 - x1
    h = y2 - y1

    xw1 = max(int(x1 - ad * w), 0)
    yw1 = max(int(y1 - ad * h), 0)
    xw2 = min(int(x2 + ad * w), img_w - 1)
    yw2 = min(int(y2 + ad * h), img_h - 1)

    face = cv2.resize(input_img[yw1:yw2+1, xw1:xw2+1,:], (64,64))
    
    data = face.transpose((2,0,1))
    data = np.expand_dims(data,axis=0)
    data = (data-127.5)/128
    data = data.astype(np.float32)
    onnx_input = {fsanet.get_inputs()[0].name: data}

    outputs = fsanet.run(None, onnx_input)
    print(outputs[0])
    yaw, pitch, roll = outputs[0][0]
    draw_axis(input_img, yaw, pitch, roll)
    cv2.imwrite('photo/test_out.png', input_img)
    return outputs

def video():
    ad = 0.6
    detector = FaceDetector()

    # capture video
    # cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture("photo/3分钟头部视频.mp4")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1024*1)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 768*1)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    img_w, img_h = 1024, 768

    onnx_model = onnxruntime.InferenceSession('model/fsanet.onnx', providers=['CPUExecutionProvider'])

    img_idx = 0
    start_time = time()
    face_time = 0
    head_time = 0
    run_time = 0
    frame_num = cap.get(7)
    
    while True:
        img_idx = img_idx + 1

        ret, input_img = cap.read()
        if not ret:
            cap.release()
            cv2.destroyAllWindows()
            print(f"fsanet平均运行时间：{run_time / frame_num}")
            break

        tik = time()
        faces = detector.predict(input_img)
        face_time = face_time + time() - tik
        tik = time()

        for face in faces:

            (x1, y1, x2, y2) = face
            w = x2 - x1
            h = y2 - y1

            xw1 = max(int(x1 - ad * w), 0)
            yw1 = max(int(y1 - ad * h), 0)
            xw2 = min(int(x2 + ad * w), img_w - 1)
            yw2 = min(int(y2 + ad * h), img_h - 1)

            data = cv2.resize(input_img[yw1:yw2+1, xw1:xw2+1,:], (64,64))
            
            data = data.transpose((2,0,1))
            data = np.expand_dims(data,axis=0)
            data = (data-127.5)/128
            data = data.astype(np.float32)
            onnx_input = {onnx_model.get_inputs()[0].name: data}
            tik = time()
            outputs = onnx_model.run(None, onnx_input)
            run_time += time() - tik
            # print(outputs[0])
            yaw, pitch, roll = outputs[0][0]
            draw_axis(input_img, yaw, pitch, roll)
        head_time = head_time + time() - tik
        tik = time()

        # cv2.imshow('result', input_img)

        # if cv2.waitKey(1) == ord('q'): # 按'q'键推出
            
        #     total_time = time() - start_time
        #     print(total_time, face_time, head_time)
        #     print(f"FPS: {img_idx / total_time}")
        #     break

if __name__ == '__main__':
    video()