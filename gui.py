import os
import sys
import cv2
# import gettext
import tkinter as tk
from tkinter import messagebox, ttk
from configparser import ConfigParser
from PIL import Image, ImageTk

from utils import win_utils
from utils.event_system import event_system
from utils.config_utils import config_utils

# windows high resolution support
import ctypes
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(1)
except:
    pass

# Internationalization
# cn = gettext.translation('visual_mouse', localedir="/language", languages=['zh_CN'])
# _ = cn.gettext

class GUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.init_data()
        self.init_ui()
        event_system.subscribe("camera_preview", self.update_frame)

    def init_data(self):
        self.entries = {}

        # self.move_type_options = {
        #     _("头部姿态->位置"): "position",
        #     _("嘴唇方向->速度"): "mouth",
        # }

        # self.click_type_option = {
        #     _("张嘴点击"): "lip",
        #     _("眨眼点击"): "sys",
        # }

        # self.scroll_type_option = {
        #     _("头部俯仰"): "head",
        #     _("嘴唇左右"): "lip",
        # }
        self.win_width = 0
        self.selected_options = ["move_type" ,"click_type", "scroll_type"]
        self.check_options = ["is_flip", "startup"]
        self.is_flip = tk.BooleanVar()
        self.startup = tk.BooleanVar()

        self.move_type_option = ["position","mouth"]
        self.click_type_option = ["lip", "eye"]
        self.scroll_type_option = ["head", "lip"]

        self.protocol("WM_DELETE_WINDOW", self.on_closing)

    def init_ui(self):
        self.title("visual mouse config")
        self.resizable(False, True)
        # self.geometry("600x1000+0+0")  # 设置窗口大小和位置（左上角）

        scroll_frame = tk.Frame(self)
        scroll_frame.pack(padx=10, pady=10)

        self.video_label = tk.Label(scroll_frame)
        self.video_label.pack()

        combo_frame = tk.Frame(scroll_frame)
        combo_frame.pack(fill="x", padx=10, pady=10)

        for i, key in enumerate(self.selected_options):
            label = tk.Label(combo_frame, text=f"{key}:")

            values = getattr(self, f"{key}_option")
            combo = ttk.Combobox(combo_frame, values=values, state="readonly", name=key)
            combo.set(config_utils.get("control_arg", key) or values[0])
            combo.bind("<<ComboboxSelected>>", self.on_combobox_select)

            label.grid(row=i, column=0, padx=10, pady=1, sticky="w")
            combo.grid(row=i, column=1, padx=10, pady=1, sticky="ew")

        combo_frame.grid_columnconfigure(0, weight=1)
        combo_frame.grid_columnconfigure(1, weight=2)

        check_frame = tk.Frame(scroll_frame)
        check_frame.pack(fill="x", padx=10, pady=5)

        for i, key in enumerate(self.check_options):
            value = config_utils.get("control_arg", key)
            var = getattr(self, key)
            var.set(eval(value))
            callback = getattr(self, f"on_{key}_click")
            checkbox = tk.Checkbutton(check_frame, text=key, variable=var, command=callback, )
            checkbox.pack(anchor="w")

        # 遍历每个section，动态生成Label和Entry
        # for section in config_utils.sections():
        #     tk.Label(scroll_frame, text=section).pack(anchor='w')
        #     frame = tk.Frame(scroll_frame)
        #     frame.pack(fill="x", padx=10, pady=5)
        #     for i, (key, value) in enumerate(config_utils[section].items()):
        #         if key in self.selected_options or key in self.check_options:
        #             continue
        #         label = tk.Label(frame, text=f"{key}:")

        #         entry = tk.Entry(frame)
        #         entry.insert(0, value)

        #         label.grid(row=i, column=0, padx=10, pady=1, sticky="w")
        #         entry.grid(row=i, column=1, padx=10, pady=1, sticky="ew")

        #         self.entries[(section, key)] = entry
        #     frame.grid_columnconfigure(0, weight=1)
        #     frame.grid_columnconfigure(1, weight=2)
        # 创建确定和重置按钮
        btn_frame = tk.Frame(scroll_frame)
        btn_frame.pack(fill="x", padx=10, pady=5)
        confirm_btn = tk.Button(btn_frame, text="确定", command=self.save_config)
        reset_btn = tk.Button(btn_frame, text="重置", command=self.reset_config)
        confirm_btn.grid(row=0, column=0, padx=10, pady=5, sticky="ew")
        reset_btn.grid(row=0, column=1, padx=10, pady=5, sticky="ew")
        btn_frame.grid_columnconfigure(0, weight=1)
        btn_frame.grid_columnconfigure(1, weight=1)
    
    def on_closing(self):
        event_system.publish("exit")

    def update_frame(self, frame):
        if self.win_width == 0:
            self.win_width = self.winfo_width() - 24
            
        height = int(self.win_width * frame.shape[0] / frame.shape[1])
        frame = cv2.resize(frame, (self.win_width, height))

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame)
        if not hasattr(self, "imgtk"):
            self.imgtk = ImageTk.PhotoImage(image=image)
            self.video_label.configure(image=self.imgtk)
        else:
            self.imgtk.paste(image)
        
    def save_config(self):
        try:
            # 遍历每个section和key，将文本框中的内容保存到config
            for (section, key), entry in self.entries.items():
                config_utils.set(section, key, entry.get())

            # 将修改后的配置写入文件
            with open(config_utils.get_config_path(), 'w') as configfile:
                config_utils.write(configfile)

            messagebox.showinfo("Success", "Configuration saved successfully!")
            event_system.publish("restart")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save configuration: {e}")

    def reset_config(self):
        # 重置文本框中的内容为配置文件中的原始值
        config.read(config_utils.get_config_path())

        for (section, key), entry in self.entries.items():
            entry.delete(0, tk.END)  # 清空文本框
            entry.insert(0, config_utils.get(section, key))  # 插入配置文件中的原始值

    def on_combobox_select(self, event):
        value = event.widget.get()
        config_utils.set("control_arg", event.widget._name, value)

    def on_is_flip_click(self):
        config_utils.set("control_arg", "is_flip", str(self.is_flip.get()))

    def on_startup_click(self):
        is_startup = self.startup.get()
        config_utils.set("control_arg", "startup", str(is_startup))
        program_name = "VisualMouse"
        program_path = os.path.abspath(sys.argv[0])
        win_utils.set_startup_program(program_name, program_path, is_startup)
        # print(program_path)

if __name__ == "__main__":
    config = ConfigParser()
    config.read("conf.ini")
    app = GUI(config)
    app.mainloop()