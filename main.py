
import cv2
import pynput
from queue import Queue

from visual_demo_v2 import VisualModule
from control_utils import ControlModule

def EXIT():
    visual_module.EXIT()
    control_module.EXIT()
    # exit()

def on_release(self, key):
    if key == pynput.keyboard.Key.esc:
        EXIT()

if __name__ == '__main__':

    keyboard_listener = pynput.keyboard.Listener(
        on_release=on_release)
    message_queue = Queue(maxsize=100)
    visual_module = VisualModule(message_queue)
    control_module = ControlModule(message_queue)
