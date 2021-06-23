# #%%
from torch import uint8

from train import *
import matplotlib.pyplot as plt
import time
from dataset import *

if param.mode == "Train":

    path_weight, path_img, path_noise = get_path()

    Gs, Ds = init_models(param.mode)
    optimizerGs, optimizerDs = init_optimizer(Gs, Ds)
    # print(Gs)

    for scale_num in range(param.num_scale):  # 0~8
        if scale_num == 0:
            pass
        else:
            Gs[scale_num].load_state_dict(
                torch.load(path_weight + "/{}_scale/netG.pth".format(scale_num - 1))
            )
            Ds[scale_num].load_state_dict(
                torch.load(path_weight + "/{}_scale/netD.pth".format(scale_num - 1))
            )
        Gs, Ds = train_single(
            Gs,
            Ds,
            scale_num,
            optimizerGs,
            optimizerDs,
            path_weight,
            path_img,
            path_noise,
        )

elif param.mode == "Test":

    start = time.time()  # 시작 시간 저장

    for imgs in test_loader:
        fixed_noise = torch.load("./noise/" + param.folder_des + "/fixed_noise.pth")
        fixed_shape = fixed_noise.shape
        fixed_noise = fixed_noise.unsqueeze(0)
        Gs, Ds = init_models(param.mode)

        for scale_num in range(param.num_scale):  # 0~7
            Gs[scale_num].load_state_dict(
                torch.load(
                    "./weights/"
                    + param.folder_des
                    + "/{}_scale/netG.pth".format(scale_num)
                )
            )
            Gs[scale_num].eval()

    for i in range(param.test_num):
        test(Gs, imgs, param.num_scale, fixed_noise, i)

    print("time :", time.time() - start)  # 현재시각 - 시작시간 = 실행 시간

