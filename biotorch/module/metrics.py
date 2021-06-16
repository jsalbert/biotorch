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
            layers_alignment[key_name] = angle
        if len(layer._modules.keys()) > 0:
            # Reverse list as we are appending from the left side of the queue
            for key in list(layer._modules.keys())[::-1]:
                queue.appendleft((layer, key))

    return layers_alignment
