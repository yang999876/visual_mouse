import cv2
from time import time

def slow_function():
    img_idx = 0
    start_time = time()
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        img_idx = img_idx + 1
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        cv2.imshow('test', cv2.flip(image, 1))
        if cv2.waitKey(1) & 0xFF == 27:
            print(f"slow function fps: {(img_idx/(time()-start_time)):.2f}")
            cv2.destroyAllWindows()
            break
    cap.release()

def fast_function():
    cap = cv2.VideoCapture(0)
    img_idx = 0
    start_time = time()
    while cap.isOpened():
        img_idx = img_idx + 1
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue
            
        cv2.imshow('MediaPipe Face Mesh', cv2.flip(image, 1))
        if cv2.waitKey(1) & 0xFF == 27:
            cv2.destroyAllWindows()
            print(f"fast function fps: {(img_idx / (time() - start_time)):2f}")
            break
    cap.release()

if __name__ == '__main__':
    slow_function()
    fast_function()