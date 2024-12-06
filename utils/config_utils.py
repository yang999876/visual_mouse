import configparser

class ConfigUtils(configparser.ConfigParser):
    def __init__(self):
        super().__init__()
        self.path = None

    def set_config_path(self, path):
        self.path = path
        self.read(self.path)

    def get_config_path(self):
        return self.path

config_utils = ConfigUtils()
