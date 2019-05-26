import os
import sys
import time
import scipy
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image


def stitch_images(inputs, *outputs, img_per_row=2):
    gap = 5
    columns = len(outputs) + 1

    width, height = inputs[0][:, :, 0].shape
    img = Image.new('RGB', (width * img_per_row * columns + gap * (img_per_row - 1), height * int(len(inputs) / img_per_row)))
    images = [inputs, *outputs]

    for ix in range(len(inputs)):
        xoffset = int(ix % img_per_row) * width * columns + int(ix % img_per_row) * gap
        yoffset = int(ix / img_per_row) * height

        for cat in range(len(images)):
            im = np.array((images[cat][ix]).cpu()).astype(np.uint8).squeeze()
            im = Image.fromarray(im)
            img.paste(im, (xoffset + cat * width, yoffset))

    return img


def visualize_tensor(tensor, imtype=np.uint8, denormtype=0, num_channel=(0, 1), path='checkpoints/visuals/', abort=True):
    """Visualize a tensor.
        Arguments:
            tensor: Input tensor
            imtype: Digit type, np.uint8, np.float64
            denormtype: Denormalization type, 0 (no denormlization, heatmap), 1 ([-1,1] to [0,255]), 2 ([0,1] to [0,255])
            num_output: if visualize heatmap choose which channels to operate,
                        if visualize image choose which samples in batch to operate,
                        default: (0,1), E.g.(5,35) is desirable if number >= 35
        """
    print("Tensor to be visualized...Shape:", tensor.shape)
    tensor = tensor.detach().cpu().float()
    if tensor.dim() == 4 and (tensor.shape[1] == 3 or (tensor.shape[1] == 1 and denormtype != 0)):
        # shape as [b, 3, h, w] or [b, 1, h, w]
        tensor = tensor.permute(0, 2, 3, 1)
        if denormtype == 2:
            tensor = tensor * 255.0
        elif denormtype == 1:
            tensor = (tensor + 1.) / 2.0 * 255.0
        else:
            assert False, "Visualize Error: Visualize image tensor with wrong de-normalization type!"
        for i in range(num_channel[0], num_channel[1]):
            image_numpy = tensor[i,:,:,:].numpy()
            image_numpy = np.clip(image_numpy, 0, 255)
            if image_numpy.shape[2] == 1:
                image_numpy = image_numpy[:, :, 0]
            image_numpy = image_numpy.astype(imtype)
            image = Image.fromarray(image_numpy)
            name = os.path.join(path, 'sample_'+ str(i) + '_' + str(int(time.time() * 100)) + '.png')
            print('\nsaving ' + name)
            image.save(name)
    elif tensor.dim() == 4:
        # shape as [b, c, h, w]
        tensor = tensor[0]
        assert denormtype != 1 and denormtype != 2, "Visualize Error: Visualize heatmap tensor with wrong de-normalization type!"

        for i in range(num_channel[0], num_channel[1]):
            image_numpy = tensor[i,:,:].numpy()
            if denormtype == 2:
                image_numpy = image_numpy * 255.0
                image_numpy = np.clip(image_numpy, 0, 255)
                image_numpy = image_numpy.astype(np.uint8)
                image = Image.fromarray(image_numpy)
                name = os.path.join(path, 'channel_'+ str(i) + '_' + str(int(time.time() * 100)) + '.png')
                print('\nsaving ' + name)
                image.save(name)
            else:
                plt.matshow(image_numpy, cmap='hot')
                plt.colorbar()
                plt.savefig(os.path.join(path, 'channel_'+ str(i) + '_' + str(int(time.time() * 100)) + '.png'))

    elif tensor.dim() == 2:
        image_numpy = tensor.numpy()
        if denormtype == 2 or denormtype == 1:
            if denormtype == 2:
                image_numpy = image_numpy * 255.0
            else:
                image_numpy = (image_numpy + 1.) / 2.0 * 255.0
            image_numpy = np.clip(image_numpy, 0, 255)
            image_numpy = image_numpy.astype(np.uint8)
            image = Image.fromarray(image_numpy)
            name = os.path.join(path, 'tensor_' + str(int(time.time() * 100)) + '.png')
            print('\nsaving ' + name)
            image.save(name)
        else:
            plt.matshow(image_numpy, cmap='hot')
            plt.colorbar()
            name = os.path.join(path, 'tensor_' + str(int(time.time() * 100)) + '.png')
            print('\nsaving ' + name)
            plt.savefig(name)

    else:
        assert False, "Visualize Error: tensor is not with valid shape!"
    if abort:
        print("Abort!")
        sys.exit(0)


class Progbar(object):
    """Displays a progress bar.
    Arguments:
        target: Total number of steps expected, None if unknown.
        width: Progress bar width on screen.
        verbose: Verbosity mode, 0 (silent), 1 (verbose), 2 (semi-verbose)
        stateful_metrics: Iterable of string names of metrics that
            should *not* be averaged over time. Metrics in this list
            will be displayed as-is. All others will be averaged
            by the progbar before display.
        interval: Minimum visual progress update interval (in seconds).
    """

    def __init__(self, target, width=25, verbose=1, interval=0.05,
                 stateful_metrics=None):
        self.target = target
        self.width = width
        self.verbose = verbose
        self.interval = interval
        if stateful_metrics:
            self.stateful_metrics = set(stateful_metrics)
        else:
            self.stateful_metrics = set()

        self._dynamic_display = ((hasattr(sys.stdout, 'isatty') and
                                  sys.stdout.isatty()) or
                                 'ipykernel' in sys.modules or
                                 'posix' in sys.modules)
        self._total_width = 0
        self._seen_so_far = 0
        # We use a dict + list to avoid garbage collection
        # issues found in OrderedDict
        self._values = {}
        self._values_order = []
        self._start = time.time()
        self._last_update = 0

    def update(self, current, values=None):
        """Updates the progress bar.
        Arguments:
            current: Index of current step.
            values: List of tuples:
                `(name, value_for_last_step)`.
                If `name` is in `stateful_metrics`,
                `value_for_last_step` will be displayed as-is.
                Else, an average of the metric over time will be displayed.
        """
        values = values or []
        for k, v in values:
            if k not in self._values_order:
                self._values_order.append(k)
            if k not in self.stateful_metrics:
                if k not in self._values:
                    self._values[k] = [v * (current - self._seen_so_far),
                                       current - self._seen_so_far]
                else:
                    self._values[k][0] += v * (current - self._seen_so_far)
                    self._values[k][1] += (current - self._seen_so_far)
            else:
                self._values[k] = v
        self._seen_so_far = current

        now = time.time()
        info = ' - %.0fs' % (now - self._start)
        if self.verbose == 1:
            if (now - self._last_update < self.interval and
                    self.target is not None and current < self.target):
                return

            prev_total_width = self._total_width
            if self._dynamic_display:
                sys.stdout.write('\b' * prev_total_width)
                sys.stdout.write('\r')
            else:
                sys.stdout.write('\n')

            if self.target is not None:
                numdigits = int(np.floor(np.log10(self.target))) + 1
                barstr = '%%%dd/%d [' % (numdigits, self.target)
                bar = barstr % current
                prog = float(current) / self.target
                prog_width = int(self.width * prog)
                if prog_width > 0:
                    bar += ('=' * (prog_width - 1))
                    if current < self.target:
                        bar += '>'
                    else:
                        bar += '='
                bar += ('.' * (self.width - prog_width))
                bar += ']'
            else:
                bar = '%7d/Unknown' % current

            self._total_width = len(bar)
            sys.stdout.write(bar)

            if current:
                time_per_unit = (now - self._start) / current
            else:
                time_per_unit = 0
            if self.target is not None and current < self.target:
                eta = time_per_unit * (self.target - current)
                if eta > 3600:
                    eta_format = '%d:%02d:%02d' % (eta // 3600,
                                                   (eta % 3600) // 60,
                                                   eta % 60)
                elif eta > 60:
                    eta_format = '%d:%02d' % (eta // 60, eta % 60)
                else:
                    eta_format = '%ds' % eta

                info = ' - ETA: %s' % eta_format
            else:
                if time_per_unit >= 1:
                    info += ' %.0fs/step' % time_per_unit
                elif time_per_unit >= 1e-3:
                    info += ' %.0fms/step' % (time_per_unit * 1e3)
                else:
                    info += ' %.0fus/step' % (time_per_unit * 1e6)

            for k in self._values_order:
                info += ' - %s:' % k
                if isinstance(self._values[k], list):
                    avg = np.mean(self._values[k][0] / max(1, self._values[k][1]))
                    if abs(avg) > 1e-3:
                        info += ' %.4f' % avg
                    else:
                        info += ' %.4e' % avg
                else:
                    info += ' %s' % self._values[k]

            self._total_width += len(info)
            if prev_total_width > self._total_width:
                info += (' ' * (prev_total_width - self._total_width))

            if self.target is not None and current >= self.target:
                info += '\n'

            sys.stdout.write(info)
            sys.stdout.flush()

        elif self.verbose == 2:
            if self.target is None or current >= self.target:
                for k in self._values_order:
                    info += ' - %s:' % k
                    avg = np.mean(self._values[k][0] / max(1, self._values[k][1]))
                    if avg > 1e-3:
                        info += ' %.4f' % avg
                    else:
                        info += ' %.4e' % avg
                info += '\n'

                sys.stdout.write(info)
                sys.stdout.flush()

        self._last_update = now

    def add(self, n, values=None):
        self.update(self._seen_so_far + n, values)


class ParamsCounter():
    def __init__(self, in_details=True):
        super(ParamsCounter, self).__init__()
        self.in_details = in_details

    def profile(self, model):

        def add_hooks(m):
            if len(list(m.children())) > 0:
                return

            m.register_buffer('total_params', torch.zeros(1))

            # compute number of parameters
            for param in m.parameters():
                m.total_params += torch.Tensor([param.numel()])

        model.apply(add_hooks)

        total_params = 0
        for m in model.modules():
            if len(list(m.children())) > 0:  # skip for non-leaf module
                continue
            if self.in_details:
                print(m.__class__.__name__ + ": " + str(int(m.total_params.item())) + " params")
            total_params += m.total_params

        total_params = total_params.item()
        print(model.__class__.__name__ + ": " + str(total_params/1e6) + "M params")
