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
    device = win32api.EnumDisplayDevices()
    settings = win32api.EnumDisplaySettings(device.DeviceName, -1)
    return settings.DisplayFrequency
    
if __name__ == '__main__':
    real_resolution = get_real_resolution()
    screen_size = get_screen_size()
    print(real_resolution)
    print(screen_size)

    screen_scale_rate = round(real_resolution[0] / screen_size[0], 2)
    print(screen_scale_rate)
