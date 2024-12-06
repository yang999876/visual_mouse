import os
import winreg
from win32 import win32api, win32gui, win32print
from win32.lib import win32con

from win32.win32api import GetSystemMetrics

def get_real_resolution():
    """获取真实的分辨率"""
    hDC = win32gui.GetDC(0)
    # 横向分辨率
    w = win32print.GetDeviceCaps(hDC, win32con.DESKTOPHORZRES)
    # 纵向分辨率
    h = win32print.GetDeviceCaps(hDC, win32con.DESKTOPVERTRES)
    return w, h

def get_screen_size():
    """获取缩放后的分辨率"""
    w = GetSystemMetrics (0)
    h = GetSystemMetrics (1)
    return w, h

def get_screen_scale_rate():
    """获取屏幕缩放率"""
    return round(get_real_resolution()[0] / get_screen_size()[0], 2)

def get_display_frequency():
    """获取屏幕刷新率"""
    device = win32api.EnumDisplayDevices()
    settings = win32api.EnumDisplaySettings(device.DeviceName, -1)
    return settings.DisplayFrequency

def set_startup_program(program_name, program_path, is_startup):
    """添加开机自启动程序"""
    startup_path = r"Software\Microsoft\Windows\CurrentVersion\Run"
    key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, startup_path, 0, winreg.KEY_SET_VALUE)
    if is_startup:
        winreg.SetValueEx(key, program_name, 0, winreg.REG_SZ, program_path)
    else:
        winreg.DeleteValue(key, program_name)
    winreg.CloseKey(key)
    return True


if __name__ == '__main__':
    print(get_real_resolution())
    print(get_screen_size())
    print(get_screen_scale_rate())
    print(get_display_frequency())
