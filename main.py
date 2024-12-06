import os
import sys
from gui import GUI
from visual_module import VisualModule
from control_module import ControlModule
from utils.event_system import event_system
from utils.config_utils import config_utils

def EXIT():
    print("exit")
    event_system.EXIT()
    gui.destroy()
    control_module.EXIT()
    visual_module.EXIT()

if __name__ == '__main__':
    root_path = os.sep.join(os.path.abspath(sys.argv[0]).split(os.sep)[:-1])

    log_path = os.path.join(root_path, "output.log")
    log_file = open(log_path, 'w')
    sys.stdout = log_file
    sys.stderr = log_file

    config_path = os.path.join(root_path, "conf.ini")
    config_utils.set_config_path(config_path)

    event_system.subscribe("exit", EXIT)

    visual_module = VisualModule()
    control_module = ControlModule()
    gui = GUI()
    gui.mainloop()
