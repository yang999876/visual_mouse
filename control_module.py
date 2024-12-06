import pynput
from threading import Thread
from time import time, sleep

from utils import win_utils
from utils.config_utils import config_utils
from utils.event_system import event_system
import pdb

class ControlModule():
    def __init__(self):
        self.init_arg()

        self.mouse = pynput.mouse.Controller()
        self.mouse_position = self.mouse.position

        event_system.subscribe("control_info", self.control_message_analyzer)
        event_system.subscribe("restart", self.init_arg)
        
        Thread(target=self.gameplay_loop, name='control_module_gameplay_loop').start()

    def init_arg(self):
        self.is_EXIT = False
        self.resolution = win_utils.get_screen_size()

        control_arg = config_utils["control_arg"]
        
        self.target_frame_rate          = eval(control_arg["target_frame_rate"])
        self.frame_intervel             = 1 / self.target_frame_rate
        self.move_speed                 = eval(control_arg["move_speed"]) # pixel/s
        self.smooth_arg                 = eval(control_arg["smooth_arg"])
        self.safe_border                = eval(control_arg["safe_border"])
        self.horizontal_max             = eval(control_arg["horizontal_max"])
        self.vertical_max               = eval(control_arg["vertical_max"])
        self.lip_direction_max          = eval(control_arg["lip_direction"])
        self.left_click_threshold       = eval(control_arg["left_click_threshold"])
        self.right_click_threshold      = eval(control_arg["right_click_threshold"])
        self.big_mouth                  = eval(control_arg["big_mouth"])
        self.small_mouth                = eval(control_arg["small_mouth"])
        self.move_type                  = control_arg["move_type"]
        self.click_type                 = control_arg["click_type"]
        self.scroll_type                = control_arg["scroll_type"]
        self.position_control_dead_zone = 50

        # internal argument
        self.vertical_current_speed     = 0 # pixel/s
        self.horizontal_current_speed   = 0
        self.vertical_target_speed      = 0
        self.horizontal_target_speed    = 0
        self.mouth_open_frames          = 0
        self.current_right_click_cd     = 0

        self.is_left_press              = False
        self.is_right_press             = False
        self.is_up_press                = False
        self.is_down_press              = False
        self.is_left_click              = False
        self.is_right_click             = False
        self.last_frame_left_eye_close  = False
        self.last_frame_right_eye_close = False
        self.horizontal_curve = {
            "init_val": 0,
            "duration": 5,
            "start_time": 0,
            "target_val": self.move_speed,
        }
        self.vertical_curve = {
            "init_val": 0,
            "duration": 5,
            "start_time": 0,
            "target_val": self.move_speed,
        }
        self.calibration_points = {
            "left_top": (self.horizontal_max, self.vertical_max),
            "right_bottom": (-self.horizontal_max, -self.vertical_max),
        }

    def debug(self):
        self.keyboard_listener = pynput.keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        self.keyboard_listener.start()

    def EXIT(self):
        print(f'control module stop')
        self.stop_control()
        self.is_EXIT = True

        if hasattr(self, "keyboard_listener"):
            self.keyboard_listener.stop()

    def stop_control(self):
        self.stop_move()
        self.left_release()
        self.right_release()

    def gameplay_loop(self):
        while not self.is_EXIT:
            start_time = time()

            self.move_curve()
            
            end_time = time()
            intervel = start_time - end_time
            waitting_time = self.frame_intervel - intervel
            if (waitting_time > 0):
                sleep(waitting_time)

    def control_message_analyzer(self, control_message):
        head_pose = control_message.get("head_pose")
        if not head_pose:
            self.stop_control()
            return

        # print(f"head pose: {head_pose}, left: {is_left_eye_closed}, right: {is_right_eye_closed}")
        self.move_logic(control_message)
        self.click_logic(control_message)
        self.scroll_logic(control_message)

    def move_logic(self, control_message):
        head_pose = control_message.get("head_pose")
        lip_direction = control_message.get("lip_direction")
        mouth_AR = control_message.get("mouth_AR")
        lip_AR = control_message.get("lip_AR")

        if self.move_type == "speed":
            if head_pose[0] < -self.horizontal_max:
                self.move_right()
            elif head_pose[0] >= -self.horizontal_max and head_pose[0] <= self.horizontal_max:
                self.stop_move_left()
                self.stop_move_right()
            elif head_pose[0] > self.horizontal_max:
                self.move_left()

            if head_pose[1] < -self.vertical_max:
                self.move_down()
            elif head_pose[1] >=-self.vertical_max and head_pose[1] <= self.horizontal_max:
                self.stop_move_up()
                self.stop_move_down() 
            elif head_pose[1] > self.horizontal_max:
                self.move_up()
        elif self.move_type == "position":
            # x_pos = (head_pose[0] + self.calibration_points["x_bias"]) * self.calibration_points["x_scalar"] * self.resolution[0]
            # y_pos = (head_pose[1] + self.calibration_points["y_bias"]) * self.calibration_points["y_scalar"] * self.resolution[1]
            x_bias = -self.calibration_points["left_top"][0]
            x_scalar = 1 / (self.calibration_points["right_bottom"][0] - self.calibration_points["left_top"][0])
            y_bias = -self.calibration_points["left_top"][1]
            y_scalar = 1 / (self.calibration_points["right_bottom"][1] - self.calibration_points["left_top"][1])

            x_pos = (head_pose[0] + x_bias) * x_scalar * self.resolution[0]
            y_pos = (head_pose[1] + y_bias) * y_scalar * self.resolution[1]
            x_move = x_pos - self.mouse.position[0]
            y_move = y_pos - self.mouse.position[1]
            # print(x_move, y_move, end="\r")
            self.horizontal_target_speed = x_move
            self.vertical_target_speed = y_move
        elif self.move_type == "mouth":
            if mouth_AR <= 0.1: # 张嘴时不移动鼠标
                if lip_direction < -self.lip_direction_max:
                    self.move_left()
                elif lip_direction > self.lip_direction_max:
                    self.move_right()
                else:
                    self.stop_move_left()
                    self.stop_move_right()

                if lip_AR < self.small_mouth:
                    self.move_down()
                elif lip_AR > self. big_mouth:
                    self.move_up()
                else:
                    self.stop_move_up()
                    self.stop_move_down() 
        else:
            raise ValueError("move_type is NOT a supported value")
    
    def click_logic(self, control_message):
        if self.click_type == "eye":

            l_EAR = control_message.get("l_EAR")
            r_EAR = control_message.get("r_EAR")
            is_right_eye_closed = True if r_EAR<0.18 else False
            is_left_eye_closed = True if l_EAR<0.18 else False

            if is_left_eye_closed:
                if not self.last_frame_left_eye_close and not is_right_eye_closed:
                    self.left_press()
                self.last_frame_left_eye_close = True
            else:
                self.left_release()
                self.last_frame_left_eye_close = False

            if is_right_eye_closed:
                if not self.last_frame_right_eye_close and not is_left_eye_closed:
                    self.right_press()
                self.last_frame_right_eye_close = True
            else:
                self.right_release()
                self.last_frame_right_eye_close = False

        elif self.click_type == "lip":

            mouth_AR = control_message.get("mouth_AR")

            if self.current_right_click_cd > 0:
                self.current_right_click_cd -= 1
            else:
                if mouth_AR > self.small_mouth:
                    self.mouth_open_frames += 1
                    if self.mouth_open_frames > self.right_click_threshold:
                        self.mouse.click(pynput.mouse.Button.right)
                        self.mouth_open_frames = 0
                        self.current_right_click_cd = self.right_click_threshold
                else:
                    if self.mouth_open_frames > 0 and self.mouth_open_frames < self.left_click_threshold:
                        self.mouse.click(pynput.mouse.Button.left)
                    self.mouth_open_frames = 0

        else:
            raise ValueError("click_type is NOT a supported value")

    def scroll_logic(self, control_message):
        if self.scroll_type == "lip":

            lip_direction = control_message.get("lip_direction")

            if lip_direction < -self.lip_direction_max:
                self.scroll_down()
            elif lip_direction > self.lip_direction_max:
                self.scroll_up()

        elif self.scroll_type == "head":

            head_pose = control_message.get("head_pose")

            if head_pose[1] < -self.vertical_max:
                self.scroll_down()
            elif head_pose[1] > self.horizontal_max:
                self.scroll_up()

        else:
            raise ValueError("click_type is NOT a supported value")

    # obsolete
    def calibration(self, control_message):
        # calibration
        # if not self.calibration_points.get("done"):
        #     if not self.calibration_points.get("left_top"):
        #         print("请盯住屏幕左上角，然后张嘴", end="\r")
        #         if not is_mouth_closed:
        #             self.calibration_points["left_top"] = head_pose
        #             self.calibration_points["dead_zone_time"] = time()
        #             print("\n")
        #     # elif not self.calibration_points.get("right_top"):
        #     #     print("请盯住屏幕右上角，然后张嘴", end="\r")
        #     #     if not is_mouth_closed and (time() - self.calibration_points["dead_zone_time"] > 1):
        #     #         self.calibration_points["right_top"] = head_pose
        #     #         self.calibration_points["dead_zone_time"] = time()
        #     #         print("\n")
        #     # elif not self.calibration_points.get("left_bottom"):
        #     #     print("请盯住屏幕左下角，然后张嘴", end="\r")
        #     #     if not is_mouth_closed and (time() - self.calibration_points["dead_zone_time"] > 1):
        #     #         self.calibration_points["left_bottom"] = head_pose
        #     #         self.calibration_points["dead_zone_time"] = time()
        #     #         print("\n")
        #     elif not self.calibration_points.get("right_bottom"):
        #         print("请盯住屏幕右下角，然后张嘴", end="\r")
        #         if not is_mouth_closed and (time() - self.calibration_points["dead_zone_time"] > 1):
        #             self.calibration_points["right_bottom"] = head_pose
        #             self.calibration_points["dead_zone_time"] = time()
        #             print("\n")
        #             print("校准完成，张嘴开始程序")
        #             print(self.calibration_points)
        #     else:
        #         self.calibration_points["done"] = True
        #         x_bias = -self.calibration_points["left_top"][0]
        #         x_scalar = 1 / (self.calibration_points["right_bottom"][0] - self.calibration_points["left_top"][0])
        #         y_bias = -self.calibration_points["left_top"][1]
        #         y_scalar = 1 / (self.calibration_points["right_bottom"][1] - self.calibration_points["left_top"][1])
        #         self.calibration_points["x_bias"] = x_bias
        #         self.calibration_points["x_scalar"] = x_scalar
        #         self.calibration_points["y_bias"] = y_bias
        #         self.calibration_points["y_scalar"] = y_scalar
        #         # print(self.calibration_points)
        #         # print((self.calibration_points["right_bottom"][0] + self.calibration_points["x_bias"]) * self.calibration_points["x_scalar"])
        #         # pdb.set_trace()
        pass

    def mouse_move_by_speed(self, vertical_speed, horizontal_speed):
        if vertical_speed == 0 and horizontal_speed == 0:
            return
        # else:
            # print(vertical_speed, horizontal_speed)
        self.mouse_position = (
            self.clamp(self.mouse_position[0] + horizontal_speed / self.target_frame_rate, 
                self.safe_border, self.resolution[0] - self.safe_border), 
            self.clamp(self.mouse_position[1] + vertical_speed / self.target_frame_rate, 
                self.safe_border, self.resolution[1] - self.safe_border)
            )
        self.mouse.position = self.mouse_position
        self.mouse_position = self.mouse.position

    def on_press(self, key):
        if key == pynput.keyboard.Key.up:
            self.move_up()
        elif key == pynput.keyboard.Key.down:
            self.move_down()
        elif key == pynput.keyboard.Key.left:
            self.move_left()
        elif key == pynput.keyboard.Key.right:
            self.move_right()
        elif key == pynput.keyboard.Key.enter:
            self.left_press()
        else:
            pass

    def on_release(self, key):
        if key == pynput.keyboard.Key.esc:
            self.EXIT()
        if key == pynput.keyboard.Key.up:
            self.stop_move_up()
        elif key == pynput.keyboard.Key.down:
            self.stop_move_down()
        elif key == pynput.keyboard.Key.left:
            self.stop_move_left()
        elif key == pynput.keyboard.Key.right:
            self.stop_move_right()
        elif key == pynput.keyboard.Key.enter:
            self.left_release()
        else:
            pass

    def left_press(self):
        if not self.is_left_click:
            self.is_left_click = True
            self.mouse.press(pynput.mouse.Button.left)

    def right_press(self):
        if not self.is_right_click:
            self.is_right_click = True
            self.mouse.press(pynput.mouse.Button.right)

    def left_release(self):
        if self.is_left_click:
            self.is_left_click = False
            self.mouse.release(pynput.mouse.Button.left)

    def right_release(self):
        if self.is_right_click:
            self.is_right_click = False
            self.mouse.release(pynput.mouse.Button.right)

    def move_up(self):
        self.stop_move_vertical()
        if not self.is_up_press:
            self.is_up_press = True
            self.vertical_target_speed -= self.move_speed
            self.vertical_curve["start_time"] = time()

    def move_down(self):
        self.stop_move_vertical()
        if not self.is_down_press:
            self.is_down_press = True
            self.vertical_target_speed += self.move_speed
            self.vertical_curve["start_time"] = time()

    def move_left(self):
        self.stop_move_horizontal()
        if not self.is_left_press:
            self.is_left_press = True
            self.horizontal_target_speed -= self.move_speed
            self.horizontal_curve["start_time"] = time()

    def move_right(self):
        self.stop_move_horizontal()
        if not self.is_right_press:
            self.is_right_press = True
            self.horizontal_target_speed += self.move_speed
            self.horizontal_curve["start_time"] = time()

    def stop_move(self):
        self.stop_move_vertical()
        self.stop_move_horizontal()

    def stop_move_vertical(self):
        self.stop_move_up()
        self.stop_move_down()
        
    def stop_move_horizontal(self):
        self.stop_move_left()
        self.stop_move_right()

    def stop_move_up(self):
        if self.is_up_press:
            self.is_up_press = False
            self.vertical_target_speed += self.move_speed

    def stop_move_down(self):
        if self.is_down_press:
            self.is_down_press = False
            self.vertical_target_speed -= self.move_speed

    def stop_move_left(self):
        if self.is_left_press:
            self.is_left_press = False
            self.horizontal_target_speed += self.move_speed    

    def stop_move_right(self):
        if self.is_right_press:
            self.is_right_press = False
            self.horizontal_target_speed -= self.move_speed    

    def scroll_down(self):
         self.mouse.scroll(dx=0, dy=-0.1)

    def scroll_up(self):
        self.mouse.scroll(dx=0, dy=0.1)

    def clamp(self, val, A, B):
        if val < A:
            return A
        if val > B:
            return B
        return val

    def lerp(self, A, B):
        output = A + (B - A) * self.smooth_arg
        if abs(output - B) < 1:
            output = B
        return output

    def square(self, initial_val, target_val, interpolate_arg):

        if interpolate_arg < 0:
            interpolate_arg = 0
        if interpolate_arg > 1:
            interpolate_arg = 1

        output = initial_val + interpolate_arg*interpolate_arg * (target_val - initial_val)

        return output

    def piecewise(self, initial_val, target_val, interpolate_arg):

        if interpolate_arg < 0:
            interpolate_arg = 0
        if interpolate_arg > 1:
            interpolate_arg = 1

        if interpolate_arg < 0.5:
            output = initial_val + 0.1 * (target_val - initial_val)
        else:
            output = target_val

        return output

    # def bezier(self, initial_val, target_val, interpolate_arg):

    def move_curve(self):
        now = time()

        if self.vertical_target_speed == 0:
            vertical_speed = 0
        else:
            # ease out
            vertical_speed = self.lerp(self.vertical_current_speed, self.vertical_target_speed)
            # vertical_interpolate_arg = \
            #     (now - self.vertical_curve["start_time"]) / self.vertical_curve["duration"]
            # vertical_speed = self.square(0, self.vertical_target_speed, vertical_interpolate_arg)

        if self.horizontal_target_speed == 0:
            horizontal_speed = 0
        else:
            horizontal_speed = self.lerp(self.horizontal_current_speed, self.horizontal_target_speed)
            # horizontal_interpolate_arg = \
            #     (now - self.horizontal_curve["start_time"]) / self.horizontal_curve["duration"]
            # horizontal_speed = self.square(0, self.horizontal_target_speed, horizontal_interpolate_arg)

        # if vertical_speed
        # print(f"vertical_speed: {vertical_speed}, horizontal_speed: {horizontal_speed}")
        self.vertical_current_speed = vertical_speed
        self.horizontal_current_speed = horizontal_speed
        self.mouse_move_by_speed(vertical_speed, horizontal_speed)

if __name__ == '__main__':
    import configparser
    config = configparser.ConfigParser()
    config.read("conf.ini")
    cm = ControlModule()
    cm.debug()