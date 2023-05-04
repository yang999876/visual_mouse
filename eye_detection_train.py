import cv2
import dlib
import h5py

import numpy as np
import tensorflow as tf

from time import time
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

from resource.face_detection import FaceDetection

# dlib的面部特征点对于侧脸识别效果很差，可能要另辟蹊径，使用整脸直接做眨眼检测
# 或者改用别的特征的检测模型

proto_path = r"resource\model\deploy.prototxt"
model_path = r"resource\model\res10_300x300_ssd_iter_140000.caffemodel"
face_detector = FaceDetection("ssd", proto_path, model_path)

predictor = dlib.shape_predictor(
    "resource/model/shape_predictor_68_face_landmarks.dat"
)

def get_eyes(img):
    img_w = img.shape[1]
    img_h = img.shape[0]
    ad = 1.6

    face = face_detector.predict(img)[0]
    face = dlib.rectangle(*face)

    shape = predictor(img, face)

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

    left_eye_left_point = shape.parts()[42]
    left_eye_right_point = shape.parts()[45]
    left_eye_middle_point = (left_eye_left_point + left_eye_right_point) / 2
    left_eye_radius = (left_eye_right_point.x - left_eye_left_point.x) / 2

    l_x1 = max(int(left_eye_middle_point.x - ad * left_eye_radius), 0)
    l_y1 = max(int(left_eye_middle_point.y - ad * left_eye_radius), 0)
    l_x2 = min(int(left_eye_middle_point.x + ad * left_eye_radius), img_w - 1)
    l_y2 = min(int(left_eye_middle_point.y + ad * left_eye_radius), img_h - 1)

    left_eye = (l_x1, l_y1, l_x2, l_y2)

    right_eye_img = img[right_eye[1]:right_eye[3], right_eye[0]:right_eye[2], :]
    left_eye_img = img[left_eye[1]:left_eye[3], left_eye[0]:left_eye[2], :]
    right_eye_img = cv2.cvtColor(right_eye_img, cv2.COLOR_BGR2GRAY)
    left_eye_img = cv2.cvtColor(left_eye_img, cv2.COLOR_BGR2GRAY)
    right_eye_img = cv2.resize(right_eye_img, (32, 32))
    left_eye_img = cv2.resize(left_eye_img, (32, 32))

    return right_eye, left_eye, right_eye_img, left_eye_img

def load_vedio(vedio_name):
    cap = cv2.VideoCapture(vedio_name)
    frame_num = cap.get(7)
    print(f'正在处理 {vedio_name} 共有 {frame_num} 帧')
    start_time = time()

    # dataset = h5py.File(f"{vedio_name}.h5", 'r+')
    # trainning_set_x = dataset['trainning_set_x']

    trainning_set_x = np.zeros((int(frame_num * 2), 32, 32), dtype='uint8')
    trainning_set_y = np.zeros((int(frame_num * 2)), dtype='bool')
    while(cap.isOpened()):
        ret, img = cap.read()
        if ret == True:
            right_eye, left_eye, right_eye_img, left_eye_img = get_eyes(img)

            frame_pos = int(cap.get(1)) * 2 - 2
            trainning_set_x[frame_pos] = left_eye_img
            trainning_set_x[frame_pos + 1] = right_eye_img

            cv2.imshow("left_eye", left_eye_img)
            cv2.imshow("right_eye", right_eye_img)

            cv2.rectangle(img, (right_eye[0], right_eye[1]), (right_eye[2], right_eye[3]), (0, 255, 0), 3)
            cv2.rectangle(img, (left_eye[0], left_eye[1]), (left_eye[2], left_eye[3]), (0, 0, 255), 3)
            cv2.imshow("image", img)
            cv2.waitKey(1)
        else:
            cap.release()
            cv2.destroyAllWindows()
            total_time = time() - start_time
            print(f'已完成：{vedio_name}，总耗时：{int(total_time)}，fps：{frame_num / total_time}')

    f = h5py.File(f'{vedio_name}.h5', 'w')
    f.create_dataset('trainning_set_x', data=trainning_set_x)
    f.create_dataset('trainning_set_y', data=trainning_set_y)
    f.close()
    # dataset.close()

# w上一张  s下一张  e\r改变标签  q退出
def tagging(dataset_path):
    dataset = h5py.File(dataset_path, 'r+')
    trainning_set_x = np.array(dataset['trainning_set_x'])
    trainning_set_y = dataset['trainning_set_y']

    frame_pos = 0
    frame_num = trainning_set_x.shape[0]
    while True:
        frame = np.zeros((40, 96), dtype='uint8')
        frame[:32, :32] = trainning_set_x[frame_pos + 1, :, :]
        frame[:32, 64:] = trainning_set_x[frame_pos, :, :]
        frame = cv2.resize(frame, (960, 400))

        tag_l = 'closed' if trainning_set_y[frame_pos + 1] else 'opened'
        tag_r = 'closed' if trainning_set_y[frame_pos] else 'opened'
        cv2.putText(frame, 
            f'{tag_l} {frame_pos +1}', 
            (0, 340), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255))
        cv2.putText(frame, 
            f'{tag_r} {frame_pos}', 
            (640, 340), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255))


        cv2.imshow('frame', frame)
        key = cv2.waitKey()
        if key == ord('q'):
            break
        elif key == ord('w'):
            frame_pos = (frame_pos - 2) % frame_num
        elif key == ord('s'):
            frame_pos = (frame_pos + 2) % frame_num
        elif key == ord('e'):
            trainning_set_y[frame_pos + 1] = not trainning_set_y[frame_pos + 1]
        elif key == ord('r'):
            trainning_set_y[frame_pos] = not trainning_set_y[frame_pos]
    dataset.close()
    cv2.destroyAllWindows()

def load_camera():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280*1)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720*1)

    net = cv2.dnn.readNetFromONNX('me.onnx')

    while True:
        ret, img = cap.read()
        right_eye, left_eye, right_eye_img, left_eye_img = get_eyes(img)

        blob_r = cv2.dnn.blobFromImage(right_eye_img, 1.0, (32, 32), (104, 117, 123), swapRB=True, crop=False)
        net.setInput(blob_r)
        right_result = net.forward()[0][0]

        blob_l = cv2.dnn.blobFromImage(left_eye_img, 1.0, (32, 32), (104, 117, 123), swapRB=True, crop=False)
        net.setInput(blob_l)
        left_result = net.forward()[0][0]

        cv2.imshow("left_eye", left_eye_img)
        cv2.imshow("right_eye", right_eye_img)

        left_color = (0, 255, 0) if left_result else (255, 0, 0)
        right_color = (0, 255, 0) if right_result else (255, 0, 0)

        cv2.rectangle(img, (right_eye[0], right_eye[1]), (right_eye[2], right_eye[3]), right_color, 3)
        cv2.rectangle(img, (left_eye[0], left_eye[1]), (left_eye[2], left_eye[3]), left_color, 3)
        cv2.imshow("image", img)
        if cv2.waitKey(1) == ord('q'): # 按'q'键推出
            break
    cv2.destroyAllWindows()

def load_dataset(file_name):
    train_dataset = h5py.File(file_name, "r")
    train_set_x_orig = np.array(train_dataset['trainning_set_x'][:]) # your train set features
    train_set_y_orig = np.array(train_dataset['trainning_set_y'][:]) # your train set labels

    m = train_set_y_orig.shape[0]
    train_num = int(0.7 * m)

    rand_index = np.arange(m)
    np.random.shuffle(rand_index)

    test_set_x_orig = train_set_x_orig[rand_index[train_num:]]
    train_set_x_orig = train_set_x_orig[rand_index[:train_num]]

    test_set_y_orig = train_set_y_orig[rand_index[train_num:]]
    train_set_y_orig = train_set_y_orig[rand_index[:train_num]]

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig

def Lenet5():
    # Create a sequential model
    model = Sequential()

    # Add layers
    model.add(Conv2D(6, kernel_size=(5, 5), strides=(1, 1), activation='relu', input_shape=(32, 32, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(16, kernel_size=(5, 5), strides=(1, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(25, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    # Compile the model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

def train():
    X_train_orig, Y_train_orig, X_test_orig, Y_test_orig = load_dataset("resource/photo/眨眼数据集.mp4.h5")

    X_train = X_train_orig/255.
    X_test = X_test_orig/255.
    Y_train = Y_train_orig.T
    Y_test = Y_test_orig.T

    print ("number of training examples = " + str(X_train.shape[0]))
    print ("number of test examples = " + str(X_test.shape[0]))
    print ("X_train shape: " + str(X_train.shape))
    print ("Y_train shape: " + str(Y_train.shape))
    print ("X_test shape: " + str(X_test.shape))
    print ("Y_test shape: " + str(Y_test.shape))

    model = Lenet5()
    model.fit(x = X_train, y = Y_train, epochs = 50, batch_size = 32)
    # model.save("me.h5")
    model = load_model(model_path)
    tf.keras.models.save_model(model, model_path[:-3])
    return model


def convert_tf(model_path):
    model = load_model(model_path)
    tf.keras.models.save_model(model, model_path[:-3])

if __name__ == '__main__':
    # load_vedio('resource/photo/大量的眨眼数据集.mp4')
    # tagging('resource/photo/大量的眨眼数据集.mp4.h5')
    # model = train()
    # convert_tf("me.h5")
    # load_camera()
    pass