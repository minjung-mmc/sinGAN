#%%
import torch
import torch.nn as nn
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import grad as torch_grad
import os
from config import *
from network import *
import time
import torch.optim as optim


def resize(img):
    img = img.squeeze(0)
    img = img.transpose(2, 1)
    m = nn.Upsample(scale_factor=4, mode="linear", align_corners=True)
    img = m(img).transpose(2, 1)
    img = img.unsqueeze(0)
    return img


def upscale_img(img, size):
    # return F.interpolate(img, size)
    m = nn.Upsample(
        size=[round(size[0]), round(size[1])], mode="bilinear", align_corners=True
    )
    return m(img)


def img_norm(img):
    return torch.clamp((img - 0.5) * 2, -1, 1)


def calc_gradient_penalty(fake, real, netD):
    # WGAN-GP Loss
    eps = torch.rand(1).to(device)
    fake = torch.autograd.Variable(fake, requires_grad=True)

    x_hat = (eps * real) + ((1 - eps) * fake)
    D_hat = netD(x_hat)
    # print(D_hat.size())
    grad = torch_grad(
        outputs=D_hat,
        inputs=x_hat,
        grad_outputs=torch.ones(D_hat.size()).to(device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    grad_penalty = param.D_lambda * ((torch.norm(grad, 2) - 1) ** 2).sum()

    return grad_penalty


def img_denorm(img):
    return np.clip((img + 1) / 2, 0, 1)


def save_img(img, path):
    img = img.cpu().detach().squeeze(0).numpy()
    img = img_denorm(img)
    img = np.transpose(img, (1, 2, 0))
    plt.imsave(path, img)


def denorm(img):
    # print(type(img), "Here")
    # img = upscale_img(img, (10, 10))
    img = img.cpu().detach().numpy().astype(np.float32)
    img = img_denorm(img)
    img = np.transpose(img, (1, 2, 0))
    return img


def generate_noise(size, batch_size):
    noise = (
        torch.rand(1, size[0], size[1], size[2])
        .expand(batch_size, size[0], size[1], size[2])
        .to(device)
    )
    noise = img_norm(noise)
    return noise


def weights_init(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def init_models(mode):
    netGs = []
    netDs = []
    for i in range(param.num_scale):
        netG = Generator().to(device)
        netG.apply(weights_init)
        if mode == "Train":
            netG.train()
        else:
            netG.eval()

        netD = Discriminator().to(device)
        netD.apply(weights_init)
        if mode == "Train":
            netD.train()
        else:
            netD.eval()

        netGs.append(netG)
        netDs.append(netD)

    return netGs, netDs


def init_optimizer(Gs, Ds):
    optimizerGs = []
    optimizerDs = []
    for i in range(param.num_scale):
        optimizerG = optim.Adam(
            Gs[i].parameters(), param.lr, betas=(param.beta1, 0.999)
        )
        optimizerD = optim.Adam(
            Ds[i].parameters(), param.lr, betas=(param.beta1, 0.999)
        )
        optimizerGs.append(optimizerG)
        optimizerDs.append(optimizerD)
    return optimizerGs, optimizerDs


def save_network(netG, netD, scale_num, path):
    path = path + "/{}_scale".format(scale_num)
    if os.path.exists(path):
        # print("There already exists folder")
        pass
    else:
        os.mkdir(path)
    torch.save(netG.state_dict(), path + "/netG.pth")
    torch.save(netD.state_dict(), path + "/netD.pth")


def reset_grads(model, require_grad):
    for p in model.parameters():
        p.requires_grad_(require_grad)
    return model


def make_dir(path):
    if os.path.exists(path):
        # print("There already exists folder")
        pass
    else:
        os.mkdir(path)


def get_path():
    folder = time.strftime("%Y-%m-%d %H:%M", time.localtime(time.time()))
    path_weight = "./weights/" + folder
    path_img = "./results/" + folder
    path_noise = "./noise/" + folder

    make_dir(path_weight)
    make_dir(path_img)
    make_dir(path_noise)
    return path_weight, path_img, path_noise


#%%
