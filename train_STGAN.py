import os
import random
import numpy as np
import argparse
import yaml
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from utils import Progbar, ParamsCounter, stitch_images
from networks import Generator, Discriminator
from dataset import Align_Celeba
from loss import GANLoss, gradient_penalty

class Config(dict):
    def __init__(self, config_path):
        super(Config, self).__init__()
        with open(config_path, 'r') as f:
            self._yaml = f.read()
            self._dict = yaml.load(self._yaml)
            self._dict['PATH'] = os.path.dirname(config_path)

    def __getattr__(self, name):
        if self._dict.get(name) is not None:
            return self._dict[name]

        return None

    def print(self):
        print('Model configurations:')
        print('---------------------------------')
        print(self._yaml)
        print('')
        print('---------------------------------')
        print('')

class STGAN(nn.Module):
    def __init__(self, config):
        super(STGAN, self).__init__()
        self.config = config
        self.FloatTensor = torch.cuda.FloatTensor if len(config.GPU) > 0 else torch.FloatTensor
        self.ByteTensor = torch.cuda.ByteTensor if len(config.GPU) > 0 else torch.ByteTensor

        # define networks
        generator = Generator(config)
        discriminator = Discriminator(config)

        # count parameters
        pc = ParamsCounter(in_details=False)
        pc.profile(discriminator)
        pc.profile(generator)

        # initialize networks
        generator = init_net(generator, config)
        discriminator = init_net(discriminator, config)

        # load pre-trained model or initialize weight
        if config.PRETRAINED:
            if os.path.exists(config.GEN_LOAD_PATH):
                print('Loading %s generator...' % config.MODEL_NAME)
                if torch.cuda.is_available():
                    data = torch.load(config.GEN_LOAD_PATH)
                else:
                    data = torch.load(config.GEN_LOAD_PATH, map_location=lambda storage, loc: storage)
                generator.load_state_dict(data['generator'])

            if os.path.exists(config.DIS_LOAD_PATH):
                print('Loading %s discriminator...' % config.MODEL_NAME)
                if torch.cuda.is_available():
                    data = torch.load(config.DIS_LOAD_PATH)
                else:
                    data = torch.load(config.DIS_LOAD_PATH, map_location=lambda storage, loc: storage)
                discriminator.load_state_dict(data['discriminator'])


        self.add_module('netG', generator)
        self.add_module('netD', discriminator)

        self.GAN_loss = GANLoss(config.GAN_MODE, tensor=self.FloatTensor, config=config)
        self.l1_loss = nn.L1Loss()

        # set optimizer
        G_params = list(self.netG.parameters())
        D_params = list(self.netD.parameters())

        beta1, beta2 = config.BETA1, config.BETA2
        G_lr, D_lr = config.LR, config.LR

        self.optimizer_G = torch.optim.Adam(G_params, lr=G_lr, betas=(beta1, beta2))
        self.optimizer_D = torch.optim.Adam(D_params, lr=D_lr, betas=(beta1, beta2))

        self.label_mode = config.LABEL_MODE
        self.thres_int = config.THRES_INT

    def process_discriminator_one_step(self, img, label):
        self.optimizer_D.zero_grad()
        gen_label = label.clone()
        for i in range(gen_label.shape[0]):
            gen_label[i, :] = label[i, torch.randperm(label.shape[1])]

        label = label.type(self.FloatTensor)
        gen_label = gen_label.type(self.FloatTensor)
        d_losses = self.compute_discriminator_loss(img, label, gen_label)
        d_loss = sum(d_losses.values()).mean()
        d_loss.backward()
        self.optimizer_D.step()
        return d_losses

    def process_generator_one_step(self, img, label):
        self.optimizer_G.zero_grad()
        gen_label = label.clone()
        for i in range(gen_label.shape[0]):
            gen_label[i, :] = label[i, torch.randperm(label.shape[1])]

        label = label.type(self.FloatTensor)
        gen_label = gen_label.type(self.FloatTensor)
        g_losses = self.compute_generator_loss(img, label, gen_label)
        g_loss = sum(g_losses.values()).mean()
        g_loss.backward()
        self.optimizer_G.step()
        return g_losses


    def compute_generator_loss(self, img, label, gen_label):
        G_losses = {}

        _gen_label = (gen_label * 2.0 - 1.0) * self.thres_int
        _label = (label * 2.0 - 1.0) * self.thres_int

        fake_image = self.generate_fake(img, _label, _gen_label)
        self_image = self.generate_fake(img, _label, _label)
        att_fake, gan_fake, _, _ = self.discriminate(fake_image, img)

        G_losses['GAN'] = self.GAN_loss(gan_fake, True, for_discriminator=False)
        G_losses['att'] = F.binary_cross_entropy_with_logits(att_fake, gen_label) * self.config.ATT_LOSS_WEIGHT
        G_losses['rec'] = self.l1_loss(self_image, img) * self.config.REC_LOSS_WEIGHT
        return G_losses


    def compute_discriminator_loss(self, img, label, gen_label):
        D_losses = {}

        _gen_label = (gen_label * 2.0 - 1.0) * self.thres_int
        _label = (label * 2.0 - 1.0) * self.thres_int

        fake_image = self.generate_fake(img, _label, _gen_label).detach()

        att_fake, gan_fake, att_real, gan_real = self.discriminate(fake_image, img)

        D_losses['D_Fake'] = self.GAN_loss(gan_fake, False, for_discriminator=True)
        D_losses['D_Real'] = self.GAN_loss(gan_real, True, for_discriminator=True)
        D_losses['D_att'] = F.binary_cross_entropy_with_logits(att_real, label)
        if self.config.GAN_MODE == 'w':
            gp = gradient_penalty(self.netD, img, fake_image, self.FloatTensor)
        else:
            gp = gradient_penalty(self.netD, img, None, self.FloatTensor)
        D_losses['D_gp'] = gp * self.config.DIS_GP_LOSS_WEIGHT
        return D_losses


    def generate_fake(self, img, label, gen_label):
        _a = gen_label - label if self.label_mode == 'diff' else gen_label
        fake_image = self.netG(img, _a)
        return fake_image


    def discriminate(self, fake_image, real_image):

        fake_and_real = torch.cat([fake_image, real_image], dim=0)

        gan, att = self.netD(fake_and_real)

        att_fake, gan_fake, att_real, gan_real = self.divide_pred(att, gan)

        return att_fake, gan_fake, att_real, gan_real


    def divide_pred(self, att, gan):
        # Take the prediction of fake and real images from the combined batch
        # the prediction contains the intermediate outputs of multiscale GAN,
        # so it's usually a list
        if type(att) == list:
            att_fake = []
            att_real = []
            gan_fake = []
            gan_real = []
            for p in att:
                att_fake.append([tensor[:tensor.size(0) // 2] for tensor in p])
                att_real.append([tensor[tensor.size(0) // 2:] for tensor in p])
            for p in gan:
                gan_fake.append([tensor[:tensor.size(0) // 2] for tensor in p])
                gan_real.append([tensor[tensor.size(0) // 2:] for tensor in p])
        else:
            att_fake = att[:att.size(0) // 2]
            att_real = att[att.size(0) // 2:]
            gan_fake = gan[:gan.size(0) // 2]
            gan_real = gan[gan.size(0) // 2:]

        return att_fake, gan_fake, att_real, gan_real

    def update_learning_rate(self, epoch):
        new_lr = self.config.LR / (10 ** (epoch // self.config.INIT_EPOCH))
        if new_lr == self.config.LR:
            return
        for param_group in self.optimizer_D.param_groups:
            param_group['lr'] = new_lr
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = new_lr
        print('update learning rate: %f' % (new_lr))

    def sample(self, img, label, iteration, att_names, att_dicts):
        with torch.no_grad():
            gen_label = Align_Celeba.generate_test_label(label, att_names, att_dicts)
            
            label = label.type(self.FloatTensor)
            gen_label = gen_label.type(self.FloatTensor)
            
            _gen_label = (gen_label * 2.0 - 1.0) * self.thres_int
            _label = (label * 2.0 - 1.0) * self.thres_int
            fake_image = self.generate_fake(img, _label, _gen_label)
            image_per_row = 4
            sample_images = stitch_images(
                postprocess(img),
                postprocess(fake_image),
                img_per_row=image_per_row
            )
            name = os.path.join(self.config.SAMPLE_PATH, str(iteration).zfill(7) + ".png")
            print('\nsaving sample ' + name)
            sample_images.save(name)

    def save(self, iteration):
        print('\nsaving %s...\n' % self.config.MODEL_NAME)
        torch.save({
            'iteration': iteration,
            'generator': self.netG.state_dict()
        }, self.config.GEN_SAVE_PATH)

        torch.save({
            'discriminator': self.netD.state_dict()
        }, self.config.DIS_SAVE_PATH)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='./configs/STGAN.yml',
                        help='training configure path (default: ./configs/STGAN.yml)')
    args = parser.parse_args()
    config = Config(args.config_path)

    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(e) for e in config.GPU)

    if torch.cuda.is_available():
        config.DEVICE = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
    else:
        config.DEVICE = torch.device("cpu")

    # initialize random seed
    torch.manual_seed(config.SEED)
    torch.cuda.manual_seed_all(config.SEED)
    np.random.seed(config.SEED)
    random.seed(config.SEED)

    train_dataset = Align_Celeba(config, part='train')
    val_dataset = Align_Celeba(config, part='val')
    sample_iterator = val_dataset.create_iterator(config.SAMPLE_SIZE)

    att_names = list(val_dataset.att_dict.keys())
    att_dicts = val_dataset.att_dict

    log_file = config.LOG_PATH

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.BATCH_SIZE,
        num_workers=4,
        drop_last=True,
        shuffle=True
    )

    model = STGAN(config)

    epoch = 0
    max_iteration = int(float((config.MAX_ITERS)))
    max_epoch = int(float(config.EPOCH))
    iteration = 0
    total = len(train_dataset)
    num_D_per_steps = config.N_D
    keep_training = True
    g_loss = torch.zeros((1,))

    while (keep_training):
        if epoch >= max_epoch:
            break
        model.update_learning_rate(epoch)
        epoch += 1

        print('\n\nTraining epoch: %d' % epoch)
        progbar = Progbar(total, width=20, stateful_metrics=['epoch', 'iter'])

        for items in train_loader:
            img, label = cuda(*items)

            d_losses = model.process_discriminator_one_step(img, label)
            d_loss = sum(d_losses.values()).mean()

            if iteration % num_D_per_steps == 0:
                g_losses = model.process_generator_one_step(img, label)
                g_loss = sum(g_losses.values()).mean()

            iteration += 1
            if iteration >= max_iteration:
                keep_training = False
                break

            bar_logs = [
                ("l_g", g_loss.item()),
                ("l_d", d_loss.item()),
                ("epoch", epoch),
                ("iter", iteration)]

            progbar.add(len(img),
                        values=bar_logs if config.VERBOSE else [x for x in bar_logs if not x[0].startswith('l_')])

            # log model at checkpoints
            if config.LOG_INTERVAL and iteration % config.LOG_INTERVAL == 0:
                logs = [
                ("| Discriminator loss:", '{0:.6f}'.format(d_loss.item())),
                ("| Fake loss:", '{0:.6f}'.format(d_losses['D_Fake'].item())),
                ("| Real loss:", '{0:.6f}'.format(d_losses['D_Real'].item())),
                ("| GP loss:", '{0:.6f}'.format(d_losses['D_gp'].item())),
                ("| Classifier loss:", '{0:.6f}'.format(d_losses['D_att'].item())),
                ("| Generator loss:", '{0:.6f}'.format(g_loss.item())),
                ("| GAN loss:", '{0:.6f}'.format(g_losses['GAN'].item())),
                ("| Classifier loss:", '{0:.6f}'.format(g_losses['att'].item())),
                ("| reconstruction loss:", '{0:.6f}'.format(g_losses['rec'].item()))]

                logs = [
                       ("", "[" + str(epoch) + "-" + str(iteration) + "]")
                   ] + logs
                with open(log_file, 'a') as f:
                    f.write('%s\n' % ' '.join([item[0] + item[1] for item in logs]))

            # sample model at checkpoints
            if config.SAMPLE_INTERVAL and iteration % config.SAMPLE_INTERVAL == 0:
                items = next(sample_iterator)
                img, label = cuda(*items)
                model.sample(img, label, iteration, att_names, att_dicts)

            # save model at checkpoints
            if config.SAVE_INTERVAL and iteration % config.SAVE_INTERVAL == 0:
                model.save(iteration)

    print('\nEnd training....')


def init_net(net, config):
    """print the network structure and initial the network"""
    if len(config.GPU) > 0:
        assert(torch.cuda.is_available())
        net.cuda()
        if len(config.GPU) > 1:
            net = nn.DataParallel(net, config.GPU)
    net.init_weights(config.WEIGHT_INIT, config.INIT_VARIANCE)
    return net


def cuda(*args):
    return (item.to(torch.device("cuda")) for item in args)


def postprocess(img):
    img = img.permute(0, 2, 3, 1)
    img = (img + 1.) / 2.0 * 255.0
    return img


if __name__ == "__main__":
    main()
