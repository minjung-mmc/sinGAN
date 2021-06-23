import torch

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
image_path_balloon = "./original/balloons.png"


class parameters:
    def __init__(self):
        self.num_scale = 8
        self.num_epoch = 2000
        self.num_iteration = 3
        self.num_channel = 3
        self.scale_ratio = 3 / 4
        self.lr = 0.0005
        self.beta1 = 0.5
        self.D_lambda = 0.1
        self.alpha = 1000
        self.noise_amp = 0.1
        self.gamma = 0.1
        self.batch_size = 1
        self.mode = "Test"
        self.test_num = 100
        self.folder_des = "2021-06-23 19:17"  # 년도-월-일 시간


param = parameters()
