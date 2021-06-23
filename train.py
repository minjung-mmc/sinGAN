#%%
import open3d as o3d
import torch.optim as optim
from dataset import *
from utils import *
from config import *
import torch.nn as nn
import cv2
import pickle

# path_weight, path_img, path_noise = get_path()


def get_prev_fixed(Gs, imgs, fixed_noise, scale_num):
    criterion = nn.MSELoss()
    noise_ratios = []
    if scale_num == 0:
        prev = torch.tensor(0.0)
    else:
        for i in range(scale_num):  # from G0 to Gn
            if i == 0:
                noise_ratio = 1  # useless Ratio
                output = Gs[i](fixed_noise, 0)

            else:
                noise_ratio = torch.sqrt(criterion(imgs[i].to(device), prev))
                # print(noise_ratio)
                output = Gs[i](0, prev)
            # print(imgs[i + 1].shape, i)
            prev = upscale_img(output, (imgs[i + 1].shape[2], imgs[i + 1].shape[3]))
            # print(prev)
            noise_ratios.append(noise_ratio)

    return prev, noise_ratios


def get_prev_var(Gs, imgs, noise_ratios, scale_num, batch_size):
    batch_size = imgs[0].shape[0]
    if scale_num == 0:
        prev = torch.tensor(0.0)
    else:
        for i in range(scale_num):
            h, w = np.shape(imgs[i])[2:4]
            if i == 0:
                var_noise = (
                    generate_noise([1, h, w], batch_size).expand(batch_size, 3, h, w)
                    * param.noise_amp
                    # generate_noise([1, h, w], batch_size)
                    # * param.noise_amp
                    # generate_noise([3, h, w])
                    # * param.noise_amp
                )

                output = Gs[i](var_noise, 0)
            else:
                var_noise = (
                    generate_noise([3, h, w], batch_size)
                    * param.noise_amp
                    * noise_ratios[i]
                    # # generate_noise([3, h, w])
                    # # * param.noise_amp
                    # generate_noise([1, h, w], batch_size)
                    # * param.noise_amp
                    # * noise_ratios[i]
                )
                # print(
                #     "var_noise",
                #     torch.min(var_noise).item(),
                #     torch.max(var_noise).item(),
                # )
                output = Gs[i](var_noise, prev)
            prev = upscale_img(output, (imgs[i + 1].shape[2], imgs[i + 1].shape[3]))
    return prev


def train_single(
    Gs, Ds, scale_num, optimizerGs, optimizerDs, path_weight, path_img, path_noise,
):
    criterion = nn.MSELoss()

    schedulerD = optim.lr_scheduler.MultiStepLR(
        optimizer=optimizerDs[scale_num],
        milestones=[param.num_epoch * 0.75],
        gamma=param.gamma,
    )
    schedulerG = optim.lr_scheduler.MultiStepLR(
        optimizer=optimizerGs[scale_num],
        milestones=[param.num_epoch * 0.75],
        gamma=param.gamma,
    )
    for epoch in range(param.num_epoch):
        for imgs in train_loader:
            img = imgs[scale_num]
            batch_size = img.shape[0]

            ############################# Bring fixed Noise ############################
            if scale_num == 0 and epoch == 0:
                h, w = np.shape(img)[2:]
                fixed_noise = (
                    generate_noise([1, h, w], batch_size).expand(batch_size, 3, h, w)
                    * param.noise_amp
                )
                torch.save(fixed_noise[0], path_noise + "/fixed_noise.pth")
            else:
                fixed_noise = torch.load(path_noise + "/fixed_noise.pth")
                fixed_noise = fixed_noise.unsqueeze(0)
            ############################# Bring fixed Noise ############################

            ############################# Bring prev ############################
            h, w = np.shape(img)[2:4]
            Gs[scale_num].train()
            Ds[scale_num].train()

            # print(h, w)
            prev_fixed, noise_ratios = get_prev_fixed(Gs, imgs, fixed_noise, scale_num)
            prev_fixed = prev_fixed.to(device)

            prev_var = get_prev_var(Gs, imgs, noise_ratios, scale_num, batch_size)
            prev_var = prev_var.to(device)
            # print(noise_ratio)
            ##################### Generate Noise ########################
            if scale_num == 0:
                # noise_var = generate_noise([1, h, w], batch_size) * param.noise_amp
                noise_var = generate_noise([3, h, w], batch_size) * param.noise_amp
                noise_var = noise_var.to(device)
            else:
                noise_ratio = torch.sqrt(criterion(prev_fixed, imgs[scale_num]))
                # noise_var = generate_noise([3, h, w]) * param.noise_amp
                noise_var = (
                    generate_noise([3, h, w], batch_size)
                    * param.noise_amp
                    * noise_ratio
                )
                # noise_var = (
                #     generate_noise([1, h, w], batch_size)
                #     * param.noise_amp
                #     * noise_ratio
                # )
                noise_var = noise_var.to(device)
            ##################### Generate Noise ########################

            # print(type(prev_fixed))
            # show_img(prev_fixed)
            # show_img(prev_var)
            for Dstep in range(param.num_iteration):
                Ds[scale_num].zero_grad()

                errD_real = Ds[scale_num](img).to(device).sum()

                if scale_num == 0:
                    fake = Gs[scale_num](noise_var, 0).to(device)
                else:
                    fake = Gs[scale_num](noise_var, prev_var).to(device)

                errD_fake = Ds[scale_num](fake).to(device).sum()

                gradient_penalty = calc_gradient_penalty(fake, img, Ds[scale_num])

                errD = -errD_real + errD_fake + gradient_penalty
                errD.backward(retain_graph=True)
                optimizerDs[scale_num].step()

            for Gstep in range(param.num_iteration):
                Gs[scale_num].zero_grad()

                if scale_num == 0:

                    fake = Gs[scale_num](noise_var, 0).to(device)
                else:
                    fake = Gs[scale_num](noise_var, prev_var).to(device)

                errG_fake = Ds[scale_num](fake).sum()
                # errG_fake = -fake_disc.sum()
                # criterion = nn.MSELoss()

                if scale_num == 0:
                    fake_fixed = Gs[scale_num](0, fixed_noise).to(device)
                else:
                    fake_fixed = Gs[scale_num](0, prev_fixed).to(device)

                # print("here", imgs[scale_num].size(), fake_fixed.size())

                rec_loss = param.alpha * criterion(fake_fixed, imgs[scale_num])
                # rec_loss = param.alpha * torch.sqrt(criterion(fake, img))

                errG = rec_loss - errG_fake

                errG.backward(retain_graph=True)
                optimizerGs[scale_num].step()
            if epoch % 100 == 0 or epoch == (param.num_epoch - 1):
                # if batch_idx % (10 * 2 ** (scale_num)) == 0 or batch_idx == (
                #     param.data_length // batch_size
                # ):
                print(
                    "scale {}: [{}/{}] error G: {:.4f}   errorD: {:.4f}".format(
                        scale_num, epoch, param.num_epoch, errG, errD,
                    )
                )
                Gs[scale_num].eval()
                Ds[scale_num].eval()
                if scale_num == 0:
                    fig, ax = plt.subplots(1, 2, figsize=[60, 10])
                    # print(fake[0].shape)
                    # fake[0] = fake[0].permute(1, 2, 0)
                    # fake_fixed[0] = fake_fixed[0].permute(1, 2, 0)
                    ax[0].imshow(denorm(fake[0].expand(3, h, w)))
                    ax[0].set_title("fake")
                    ax[1].imshow(denorm(fake_fixed[0].expand(3, h, w)))
                    ax[1].set_title("fake_fixed")
                    # plt.show()
                    plt.savefig(path_img + "/img{}_{}".format(scale_num, epoch))
                    plt.close()
                else:
                    fig, ax = plt.subplots(1, 5, figsize=[60, 10])
                    # print(denorm(fake))
                    # fake[0] = fake[0].permute(1, 2, 0)
                    # prev_var[0] = prev_var[0].permute(1, 2, 0)
                    # fake_fixed[0] = fake_fixed[0].permute(1, 2, 0)
                    # prev_fixed[0] = prev_fixed[0].permute(1, 2, 0)
                    # imgs[scale_num][0] = imgs[scale_num][0].permute(1, 2, 0)
                    ax[0].imshow(denorm(fake[0].expand(3, h, w)))
                    ax[0].set_title("fake")
                    ax[1].imshow(denorm(prev_var[0].expand(3, h, w)))
                    ax[1].set_title("prev_var")
                    ax[2].imshow(denorm(fake_fixed[0].expand(3, h, w)))
                    ax[2].set_title("fake_fixed")
                    ax[3].imshow(denorm(prev_fixed[0].expand(3, h, w)))
                    ax[3].set_title("prev_fixed")
                    ax[4].imshow(denorm(img[0].expand(3, h, w)))
                    ax[4].set_title("real")
                    # plt.show()
                    plt.savefig(path_img + "/img{}_{}".format(scale_num, epoch))
                    plt.close("all")
        schedulerD.step()
        schedulerG.step()
        Gs[scale_num].eval()
        Ds[scale_num].eval()
        save_network(Gs[scale_num], Ds[scale_num], scale_num, path_weight)
    return Gs, Ds


def test(Gs, imgs, num_scale, fixed_noise, j):
    with torch.no_grad():
        _, noise_ratios = get_prev_fixed(Gs, imgs, fixed_noise, (num_scale - 1))

        for i in range(0, num_scale - 1):
            h, w = np.shape(imgs[i])[2:]
            var_noise = (
                generate_noise([3, h, w], param.batch_size)
                * param.noise_amp
                * noise_ratios[i]
            )
            if i == 0:
                output = Gs[i](var_noise, 0)
                # output = Gs[i](0, img)
                # output = Gs[i](noise_var, prev)
            else:
                output = Gs[i](var_noise, prev)
                # output = Gs[i](0, prev)
            prev = upscale_img(output, (imgs[i + 1].shape[2], imgs[i + 1].shape[3]))
        # print(prev.shape)
        h, w = imgs[num_scale - 1].shape[2:4]
        var_noise = (
            generate_noise([3, h, w], param.batch_size)
            * param.noise_amp
            * noise_ratios[i]
        )

        fake = Gs[num_scale - 1](var_noise, prev)

        #########################
        # return fake.squeeze(0).squeeze(0).cpu()

        #######################################save fig #############################################
        h, w = fake.shape[2:4]
        # print("dlkfjslkdj", fake.shape, h, w)
        make_dir("./test/" + param.folder_des)
        plt.imsave(
            "./test/" + param.folder_des + "/test{}.png".format(j), denorm(fake[0]),
        )
        #######################################save fig #############################################
        # print("success")


# def test(Gs, imgs, num_scale, fixed_noise, j):
#     with torch.no_grad():
#         _, noise_ratios = get_prev_fixed(Gs, imgs, fixed_noise, (num_scale - 1))
#         img = cv2.cvtColor(cv2.imread(param.testDataPath), cv2.COLOR_BGR2GRAY)
#         # img = img.convert("L")
#         img = torch.tensor(img).unsqueeze(2)
#         img = torch.transpose(img, 0, 2)
#         img = torch.transpose(img, 1, 2).unsqueeze(0)
#         img = img.type(torch.FloatTensor).to(device)
#         img = img / 255.0
#         img = resize(img)
#         img = img_norm(img)

#         h, w = img.shape[2:4]
#         var_noise = (
#             generate_noise([1, h, w], param.batch_size)
#             * param.noise_amp
#             * noise_ratios[-1]
#         )

#         fake = Gs[num_scale - 1](0, img)
#         h, w = fake.shape[2:4]
#         # print("dlkfjslkdj", fake.shape, h, w)
#         make_dir("./test/" + param.folder_des)
#         make_dir("./test/" + param.folder_des + "/{}".format(param.mcdrop))
#         plt.imsave(
#             "./test/"
#             + param.folder_des
#             + "/{}".format(param.mcdrop)
#             + "/test{}.png".format(j),
#             denorm(fake[0].expand(3, h, w)),
#         )
#         # print("success")


# denorm(prev_fixed[0].expand(3, h, w))

#%%
