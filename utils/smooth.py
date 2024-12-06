
class Smooth(object):
    def __init__(self, args_list, smooth_step = 5, smooth_type = "pow"):
        self.smooth_step = smooth_step
        self.smooth_type = smooth_type
        self.current_step = 0
        self.args = {}
        for arg in args_list:
            self.args[arg] = [0] * smooth_step

    def smooth(self, **kwargs):
        out_dict = {}
        for k,v in kwargs.items():
            arg = self.args[k]
            arg[self.current_step] = v
            out_value = 0
            for i in range(self.smooth_step):
                if self.smooth_type == "pow":
                    scalar = -i if i==self.smooth_step-1 else -i-1
                    out_value += arg[(self.current_step - i) % self.smooth_step] * pow(2, scalar)
                if self.smooth_type == "mean":
                    out_value += arg[i] / self.smooth_step
            out_dict[k] = out_value
            arg[self.current_step] = out_value
        self.current_step = (self.current_step + 1) % self.smooth_step
        return out_dict

if __name__ == '__main__':
    a = Smooth(5, ['arg1', 'arg2'])
    print(a.smooth(arg1=10, arg2=20))
    print(a.smooth(arg1=10, arg2=20))
    print(a.smooth(arg1=10, arg2=20))
    print(a.smooth(arg1=10, arg2=20))
    print(a.smooth(arg1=10, arg2=20))
    print(a.smooth(arg1=10, arg2=20))