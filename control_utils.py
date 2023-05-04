import pynput
from threading import Thread
from time import time, sleep
import win_utils
import pdb

class ControlModule():
    def __init__(self, message_queue):

        self.message_queue = message_queue
        self.is_EXIT = False
        self.target_frame_rate = 120
        self.frame_intervel = 1 / self.target_frame_rate
        self.move_speed = 400
        self.smooth_arg = 0.1
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

        self.control_type = "position"
        self.calibration_points = {
            "left_top": (-8, 5),
            "right_bottom": (8, -5),
        }
        self.resolution = win_utils.get_screen_size()
        self.save_border = 10

        self.vertical_current_speed = 0 # pixel/s
        self.horizontal_current_speed = 0
        self.vertical_target_speed = 0
        self.horizontal_target_speed = 0
        self.is_left_press = False
        self.is_right_press = False
        self.is_up_press = False
        self.is_down_press = False
        self.is_left_click = False
        self.is_right_click = False
        self.last_frame_left_eye_close = False
        self.last_frame_right_eye_close = False
        self.position_control_dead_zone = 50
        self.mouse = pynput.mouse.Controller()
        self.mouse_position = self.mouse.position

        Thread(target=self.gameplay_loop, name='control_module_gameplay_loop').start()
        Thread(target=self.control_message_get_loop, name='control_module_message_loop').start()

        self.keyboard_listener = pynput.keyboard.Listener(
            on_press=self.on_press, 
            on_release=self.on_release)
        self.keyboard_listener.start()

    def EXIT(self):
        self.is_EXIT = True
        self.stop_control()

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

    def control_message_get_loop(self):
        while not self.is_EXIT:
            control_message = self.message_queue.get()
            self.control_message_analyzer(control_message)

    def control_message_analyzer(self, control_message):
        head_pose = control_message.get("head_pose")
        if not head_pose:
            self.stop_control()
            return

        is_left_eye_closed = control_message.get("is_left_eye_closed")
        is_right_eye_closed = control_message.get("is_right_eye_closed")
        is_mouth_closed = control_message.get("is_mouth_closed")
        is_lip_closed = control_message.get("is_lip_closed")
        lip_direction = control_message.get("lip_direction")

        # print(f"head pose: {head_pose}, left: {is_left_eye_closed}, right: {is_right_eye_closed}")
        if self.control_type == "speed":

            if head_pose[0] < -8:
                # move to left
                self.move_left()
            elif head_pose[0] >= -8 and head_pose[0] <= 8:
                # keep stop
                self.stop_move_left()
                self.stop_move_right()
            elif head_pose[0] > 8:
                # move to right
                self.move_right()

            if head_pose[1] < -5:
                # move down
                self.move_down()
            elif head_pose[1] >=-5 and head_pose[1] <= 8:
                # keep stop
                self.stop_move_up()
                self.stop_move_down() 
            elif head_pose[1] > 8:
                # move up
                self.move_up()

            if lip_direction < -7.5:
                # scroll down
                self.scroll_down()
            if lip_direction > 7.5:
                # scroll up
                self.scroll_up()

            # if lip_direction < -0.1:
            #     # move right
            #     self.move_right()
            # elif lip_direction > 0.1:
            #     # move left
            #     self.move_left()
            # elif lip_direction >=-0.1 and lip_direction <= 0.1:
            #     # keep stop
            #     self.stop_move_left()
            #     self.stop_move_right() 

            #     if not is_mouth_closed:
            #         # move down
            #         self.move_down()
            #     elif is_lip_closed:
            #         # move up
            #         self.move_up()
            #     else:
            #         # stop move
            #         self.stop_move_up()
            #         self.stop_move_down()

            if is_left_eye_closed:
                if not self.last_frame_left_eye_close and not is_right_eye_closed:
                    self.left_click()
                self.last_frame_left_eye_close = True
            else:
                self.left_release()
                self.last_frame_left_eye_close = False

            if is_right_eye_closed:
                if not self.last_frame_right_eye_close and not is_left_eye_closed:
                    self.right_click()
                self.last_frame_right_eye_close = True
            else:
                self.right_release()
                self.last_frame_right_eye_close = False
        elif self.control_type == "position":
            # click logic
            if is_left_eye_closed:
                if not self.last_frame_left_eye_close and not is_right_eye_closed:
                    self.left_click()
                self.last_frame_left_eye_close = True
            else:
                self.left_release()
                self.last_frame_left_eye_close = False

            if is_right_eye_closed:
                if not self.last_frame_right_eye_close and not is_left_eye_closed:
                    self.right_click()
                self.last_frame_right_eye_close = True
            else:
                self.right_release()
                self.last_frame_right_eye_close = False
            # scroll logic
            if lip_direction < -7.5:
                # scroll down
                self.scroll_down()
            elif lip_direction > 7.5:
                # scroll up
                self.scroll_up()
                # if scroll, dont move

            # calibration
            if not self.calibration_points.get("done"):
                if not self.calibration_points.get("left_top"):
                    print("请盯住屏幕左上角，然后张嘴", end="\r")
                    if not is_mouth_closed:
                        self.calibration_points["left_top"] = head_pose
                        self.calibration_points["dead_zone_time"] = time()
                        print("\n")
                # elif not self.calibration_points.get("right_top"):
                #     print("请盯住屏幕右上角，然后张嘴", end="\r")
                #     if not is_mouth_closed and (time() - self.calibration_points["dead_zone_time"] > 1):
                #         self.calibration_points["right_top"] = head_pose
                #         self.calibration_points["dead_zone_time"] = time()
                #         print("\n")
                # elif not self.calibration_points.get("left_bottom"):
                #     print("请盯住屏幕左下角，然后张嘴", end="\r")
                #     if not is_mouth_closed and (time() - self.calibration_points["dead_zone_time"] > 1):
                #         self.calibration_points["left_bottom"] = head_pose
                #         self.calibration_points["dead_zone_time"] = time()
                #         print("\n")
                elif not self.calibration_points.get("right_bottom"):
                    print("请盯住屏幕右下角，然后张嘴", end="\r")
                    if not is_mouth_closed and (time() - self.calibration_points["dead_zone_time"] > 1):
                        self.calibration_points["right_bottom"] = head_pose
                        self.calibration_points["dead_zone_time"] = time()
                        print("\n")
                        print("校准完成，张嘴开始程序")
                        print(self.calibration_points)
                else:
                    self.calibration_points["done"] = True
                    x_bias = -self.calibration_points["left_top"][0]
                    x_scalar = 1 / (self.calibration_points["right_bottom"][0] - self.calibration_points["left_top"][0])
                    y_bias = -self.calibration_points["left_top"][1]
                    y_scalar = 1 / (self.calibration_points["right_bottom"][1] - self.calibration_points["left_top"][1])
                    self.calibration_points["x_bias"] = x_bias
                    self.calibration_points["x_scalar"] = x_scalar
                    self.calibration_points["y_bias"] = y_bias
                    self.calibration_points["y_scalar"] = y_scalar
                    print(self.calibration_points)
                    print((self.calibration_points["right_bottom"][0] + self.calibration_points["x_bias"]) * self.calibration_points["x_scalar"])
                    # pdb.set_trace()
            # movement logic
            else:
                x_pos = (head_pose[0] + self.calibration_points["x_bias"]) * self.calibration_points["x_scalar"] * self.resolution[0]
                y_pos = (head_pose[1] + self.calibration_points["y_bias"]) * self.calibration_points["y_scalar"] * self.resolution[1]
                x_move = x_pos - self.mouse.position[0]
                y_move = y_pos - self.mouse.position[1]
                print(x_move, y_move, end="\r")
                self.horizontal_target_speed = x_move
                self.vertical_target_speed = y_move
                # if x_move < -self.position_control_dead_zone:
                #     # move left
                #     self.horizontal_target_speed = -self.move_speed
                # elif x_move < self.position_control_dead_zone:
                #     self.horizontal_target_speed = 0
                # else:
                #     # move right
                #     self.horizontal_target_speed = self.move_speed

                # if y_move < -self.position_control_dead_zone:
                #     # move up
                #     self.vertical_target_speed = -self.move_speed
                # elif y_move < self.position_control_dead_zone:
                #     self.vertical_target_speed = 0
                # else:
                #     # move down
                #     self.vertical_target_speed = self.move_speed
                # if x_move < -10:
                #     # move left
                #     self.move_left()
                # elif x_move < 10:
                #     self.stop_move_horizontal()
                # else:
                #     # move right
                #     self.move_right()

                # if y_move < -10:
                #     # move up
                #     self.move_up()
                # elif y_move < 10:
                #     self.stop_move_vertical()
                # else:
                #     # move down
                #     self.move_down()
        else:
            raise ValueError("control_type is NOT a supported value")

    def mouse_move_by_speed(self, vertical_speed, horizontal_speed):
        if vertical_speed == 0 and horizontal_speed == 0:
            return
        # else:
            # print(vertical_speed, horizontal_speed)
        self.mouse_position = (
            self.limit(self.mouse_position[0] + horizontal_speed / self.target_frame_rate, 
                self.save_border, self.resolution[0] - self.save_border), 
            self.limit(self.mouse_position[1] + vertical_speed / self.target_frame_rate, 
                self.save_border, self.resolution[1] - self.save_border)
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
            self.left_click()
        else:
            pass

    def on_release(self, key):
        if key == pynput.keyboard.Key.esc:
            self.exit_gameloop()
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

    def exit_gameloop(self):
        print(f'control utils stop')
        self.keyboard_listener.stop()
        self.is_EXIT = True

    def left_click(self):
        if not self.is_left_click:
            self.is_left_click = True
            self.mouse.press(pynput.mouse.Button.left)

    def right_click(self):
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
            self.stop_move_down()
            self.is_up_press = True
            self.vertical_target_speed -= self.move_speed
            self.vertical_curve["start_time"] = time()

    def move_down(self):
        self.stop_move_vertical()
        if not self.is_down_press:
            self.stop_move_right()
            self.is_down_press = True
            self.vertical_target_speed += self.move_speed
            self.vertical_curve["start_time"] = time()

    def move_left(self):
        self.stop_move_horizontal()
        if not self.is_left_press:
            self.is_right_press = True
            self.horizontal_target_speed -= self.move_speed
            self.horizontal_curve["start_time"] = time()

    def move_right(self):
        self.stop_move_horizontal()
        if not self.is_right_press:
            self.stop_move_left()
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

    def limit(self, val, A, B):
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
    ControlModule()