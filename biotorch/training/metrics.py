import torch


from collections import deque, OrderedDict, defaultdict


def compute_angles_module(module):
    queue = deque()
    layers_alignment = OrderedDict()
    seen_keys = defaultdict(lambda: 0)

    # First pass to store module keys
    for module_keys in module._modules.keys():
        queue.append((module, module_keys))

    # Approximate depth first traversal of the model using a deque
    while len(queue) > 0:
        module, module_key = queue.popleft()
        layer = getattr(module, module_key)
        if 'alignment' in layer.__dict__:
            angle = layer.compute_alignment()
            key_name = module_key + '_' + str(seen_keys[module_key])
            seen_keys[module_key] += 1
            layers_alignment[key_name] = angle.item()
        if len(layer._modules.keys()) > 0:
            # Reverse list as we are appending from the left side of the queue
            for key in list(layer._modules.keys())[::-1]:
                queue.appendleft((layer, key))

    return layers_alignment


def compute_weight_ratio_module(module, mode):
    queue = deque()
    weight_diff = OrderedDict()
    seen_keys = defaultdict(lambda: 0)

    # First pass to store module keys
    for module_keys in module._modules.keys():
        queue.append((module, module_keys))

    # Approximate depth first traversal of the model using a deque
    while len(queue) > 0:
        module, module_key = queue.popleft()
        layer = getattr(module, module_key)
        weight = None
        if mode == 'backpropagation' and isinstance(layer, (torch.nn.Conv2d, torch.nn.Linear)):
            with torch.no_grad():
                weight = torch.linalg.norm(layer.weight)

        elif 'weight_ratio' in layer.__dict__:
            weight = layer.compute_weight_ratio()

        if weight is not None:
            key_name = module_key + '_' + str(seen_keys[module_key])
            seen_keys[module_key] += 1
            weight_diff[key_name] = weight.item()

        if len(layer._modules.keys()) > 0:
            # Reverse list as we are appending from the left side of the queue
            for key in list(layer._modules.keys())[::-1]:
                queue.appendleft((layer, key))

    return weight_diff


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
