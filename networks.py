import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm
import torchvision
import numpy as np

MAX_DIM = 64 * 16

class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    def init_weights(self, init_type='normal', gain=0.02):
        '''
        initialize network's weights
        init_type: normal | xavier | kaiming | orthogonal
        '''
        def init_func(m):
            classname = m.__class__.__name__
            # print(classname)
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                # print(m.weight.data.size())
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)
                elif init_type == 'none':  # uses pytorch's default init method
                    m.reset_parameters()
                else:
                    raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
                if hasattr(m, 'bias') and m.bias is not None:
                    # print(m.bias.data.size())
                    nn.init.constant_(m.bias.data, 0.0)

            elif classname.find('BatchNorm2d') != -1 or classname.find('InstanceNorm2d') != -1:
                if hasattr(m, 'weight') and m.weight is not None:
                    # print(m.weight.data.size())
                    nn.init.normal_(m.weight.data, 1.0, gain)
                if hasattr(m, 'bias') and m.bias is not None:
                    # print(m.bias.data.size())
                    nn.init.constant_(m.bias.data, 0.0)

        print('initialize network with %s' % init_type)
        self.apply(init_func)

class GeneratorEncoder(nn.Module):
    def __init__(self, dim=64, n_layers=5, multi_input=1):
        super(GeneratorEncoder, self).__init__()
        self.dim = dim
        self.layers = n_layers
        self.multi_input = multi_input
        d = 3
        for i in range(n_layers):
            pre_d = d
            d = min(dim * 2 ** i, MAX_DIM)
            if multi_input > i and i > 0:
                pre_d = pre_d + 3
            conv = nn.Sequential(nn.Conv2d(pre_d, d, kernel_size=4, stride=2, padding=1),
                                 nn.BatchNorm2d(d),
                                 nn.LeakyReLU(0.2))
            setattr(self, 'encoder_'+str(i), conv)

    def forward(self, x_input):
        x = x_input
        xs = []
        for i in range(self.layers):
            if self.multi_input > i and i > 0:
                x_multi = F.avg_pool2d(x_input, kernel_size=2**i, stride=2**i)
                x = torch.cat([x, x_multi], dim=1)
            model = getattr(self, 'encoder_'+str(i))
            x = model(x)
            xs.append(x)
        return xs


class GeneratorDecoder(nn.Module):
    def __init__(self, dim=64, att_dim=40, n_layers=5, shortcut_layers=1, inject_layers=0, one_more_conv=0):
        super(GeneratorDecoder, self).__init__()
        self.n_layers = n_layers
        self.dim = dim
        shortcut_layers = min(shortcut_layers, n_layers - 1)
        inject_layers = min(inject_layers, n_layers - 1)
        self.shortcut_layers = shortcut_layers
        self.inject_layers = inject_layers
        self.one_more_conv = one_more_conv
        pre_d = min(dim * 2 ** (n_layers - 1), MAX_DIM) + att_dim
        for i in range(n_layers):
            if i < n_layers - 1:
                d = min(dim * 2 ** (n_layers - 1 - i), MAX_DIM)
                deconv = nn.Sequential(nn.ConvTranspose2d(pre_d, d, kernel_size=4, stride=2, padding=1),
                                       nn.BatchNorm2d(d),
                                       nn.ReLU())
                pre_d = d
                if shortcut_layers > i:
                    pre_d = pre_d + min(dim * 2 ** (n_layers - 2 - i), MAX_DIM)
                if inject_layers > i:
                    pre_d = pre_d + att_dim
            else:
                if one_more_conv:
                    d = min(dim * 2 ** (n_layers - 1 - i), MAX_DIM) // 4
                    layers = [nn.ConvTranspose2d(pre_d, d, kernel_size=4, stride=2, padding=1),
                              nn.BatchNorm2d(d),
                              nn.ReLU()]
                    layers += [nn.ConvTranspose2d(d, 3, kernel_size=one_more_conv, stride=1, padding=1 if one_more_conv == 3 else 0),
                               nn.Tanh()]
                else:
                    d = 3
                    layers = [nn.ConvTranspose2d(pre_d, d, kernel_size=4, stride=2, padding=1),
                              nn.Tanh()]
                deconv = nn.Sequential(*layers)
            setattr(self, 'decoder_'+str(i), deconv)

    def concat(self, z, z_, _a):
        features = [z]
        if z_ is not None:
            features.append(z_)
        if _a is not None:
            _a = _a.view(-1, _a.shape[1], 1, 1)
            _a = _a.repeat(1, 1, z.shape[2], z.shape[3])
            features.append(_a)
        return torch.cat(features, dim=1)

    def forward(self, zs, _a):
        z = self.concat(zs[-1], None, _a)
        for i in range(self.n_layers):
            model = getattr(self, 'decoder_'+str(i))
            if i < self.n_layers - 1:
                z = model(z)
                if self.shortcut_layers > i:
                    z = self.concat(z, zs[self.n_layers - 2 - i], None)
                if self.inject_layers > i:
                    z = self.concat(z, None, _a)
            else:
                x = model(z)
        return x


class ConvGRUCell(nn.Module):
    def __init__(self, in_channel, state_channel, out_channel, norm='none', pass_state='lstate'):
        super(ConvGRUCell, self).__init__()
        self.pass_state = pass_state
        self.deconv = nn.ConvTranspose2d(state_channel, out_channel, 4, 2, padding=1)
        # self.deconv = nn.Sequential(nn.UpsamplingBilinear2d(),
        #                             nn.Conv2d(state_channel, out_channel, 4, 1))
        if norm == 'instance':
            self.gate = nn.Sequential(nn.Conv2d(in_channel+out_channel, out_channel, 3, 1, padding=1),
                                      nn.InstanceNorm2d(out_channel, affine=False),
                                      nn.Sigmoid())
            self.info = nn.Sequential(nn.Conv2d(in_channel+out_channel, out_channel, 3, 1, padding=1),
                                      nn.InstanceNorm2d(out_channel, affine=False),
                                      nn.Tanh())
        elif norm == 'instance':
            self.gate = nn.Sequential(nn.Conv2d(in_channel+out_channel, out_channel, 3, 1, padding=1),
                                      nn.BatchNorm2d(out_channel, affine=True),
                                      nn.Sigmoid())
            self.info = nn.Sequential(nn.Conv2d(in_channel+out_channel, out_channel, 3, 1, padding=1),
                                      nn.BatchNorm2d(out_channel, affine=True),
                                      nn.Tanh())
        else:
            self.gate = nn.Sequential(nn.Conv2d(in_channel+out_channel, out_channel, 3, 1, padding=1),
                                      nn.Sigmoid())
            self.info = nn.Sequential(nn.Conv2d(in_channel+out_channel, out_channel, 3, 1, padding=1),
                                      nn.Tanh())


    def forward(self, in_data, state):
        state_ = self.deconv(state)
        reset_gate = self.gate(torch.cat([in_data, state_], dim=1))
        update_gate = self.gate(torch.cat([in_data, state_], dim=1))
        new_state = reset_gate * state_
        new_info = self.info(torch.cat([in_data, new_state], dim=1))
        output = (1 - update_gate) * state_ + update_gate * new_info
        if self.pass_state == 'gru':
            return output, output
        elif self.pass_state == 'direct':
            return output, state_
        else: #'stu'
            return output, new_state


class GeneratorSkipConnectionUnit(nn.Module):
    def __init__(self, dim=64, att_dim=40, n_layers=4, inject_layers=0, enc_layers=5, norm='none', pass_state='stu'):
        super(GeneratorSkipConnectionUnit, self).__init__()
        self.dim = dim
        self.n_layers = n_layers
        self.inject_layers = inject_layers
        self.enc_layers = enc_layers
        assert enc_layers == n_layers + 1
        state_d = min(dim * 2 ** (enc_layers - 1), MAX_DIM) + att_dim
        for i in range(n_layers):
            input_d = min(dim * 2**(n_layers - 1 - i), MAX_DIM)
            cell = ConvGRUCell(input_d, state_d, input_d, norm, pass_state)
            setattr(self, 'cell_'+str(i), cell)
            if inject_layers > i:
                state_d = input_d + att_dim
            else:
                state_d = input_d

    def concat(self, z, z_, _a):
        features = [z]
        if z_ is not None:
            features.append(z_)
        if _a is not None:
            _a = _a.view(-1, _a.shape[1], 1, 1)
            _a = _a.repeat(1, 1, z.shape[2], z.shape[3])
            features.append(_a)
        return torch.cat(features, dim=1)

    def forward(self, zs, _a):
        zs_ = [zs[-1]]
        state = self.concat(zs[-1], None, _a)

        for i in range(self.n_layers):
            cell = getattr(self, 'cell_'+str(i))
            output = cell(zs[self.n_layers - 1 - i], state)
            zs_.insert(0, output[0])
            if self.inject_layers > i:
                state = self.concat(output[1], None, _a)
            else:
                state = output[1]
        return zs_


class Generator(BaseNetwork):
    def __init__(self, config):
        super(Generator, self).__init__()
        self.encoder = GeneratorEncoder(config.ENC_DIM, config.ENC_LAYERS, config.MULTI_INPUTS)
        self.stu = GeneratorSkipConnectionUnit(config.STU_DIM, config.ATT_DIM, config.STU_LAYERS,
                                               config.STU_INJECT_LAYERS, config.ENC_LAYERS, config.STU_NORM,
                                               config.STU_STATE)
        self.decoder = GeneratorDecoder(config.DEC_DIM, config.ATT_DIM, config.DEC_LAYERS, config.SKIP_LAYERS,
                                        config.INJECT_LAYERS)
        self.use_stu = config.USE_STU
    def forward(self, x, _a):
        xs = self.encoder(x)
        zs = self.stu(xs, _a) if self.use_stu else xs
        output = self.decoder(zs, _a)
        return output


class Discriminator(BaseNetwork):
    def __init__(self, config):
        super(Discriminator, self).__init__()
        self.n_layers = config.DIS_LAYERS
        pre_d = 3
        for i in range(config.DIS_LAYERS):
            d = min(config.DIS_DIM * 2 ** i, MAX_DIM)
            conv = nn.Sequential(nn.Conv2d(pre_d, d, kernel_size=4, stride=2, padding=1),
                                 nn.InstanceNorm2d(d, affine=True),
                                 nn.LeakyReLU(0.2))
            setattr(self, 'discriminator_'+str(i), conv)
            pre_d = d
        y_dim = (config.INPUT_SIZE[0] // (2 ** config.DIS_LAYERS)) ** 2 * pre_d
        fc_dim = config.DIS_FC_DIM
        att_dim = config.ATT_DIM
        self.logit_gan = nn.Sequential(nn.Linear(y_dim, fc_dim),
                                       nn.LeakyReLU(0.2),
                                       nn.Linear(fc_dim, 1))
        self.logit_att = nn.Sequential(nn.Linear(y_dim, fc_dim),
                                       nn.LeakyReLU(0.2),
                                       nn.Linear(fc_dim, att_dim))

    def forward(self, x):
        for i in range(self.n_layers):
            model = getattr(self, 'discriminator_'+str(i))
            x = model(x)
        y = x.view(x.shape[0], -1)
        logit_gan = self.logit_gan(y)
        logit_att = self.logit_att(y)
        return logit_gan, logit_att

if __name__ == "__main__":
    import yaml
    import argparse
    import os
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

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='./configs/STGAN.yml',
                        help='training configure path (default: ./configs/STGAN.yml)')
    args = parser.parse_args()
    config = Config(args.config_path)
    netD = Discriminator(config)
    netD.init_weights(config.WEIGHT_INIT)
    netG = Generator(config)
    netG.init_weights(config.WEIGHT_INIT)

