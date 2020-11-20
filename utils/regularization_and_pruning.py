import torch


def find_ths_by_dichotomy(target, model):
    params = torch.cat([i.weight.abs().flatten() for n, i in model.named_modules() if 'prunable' in n]).sort()[0]
    max = len(torch.cat([i.abs().flatten() for i in model.parameters()]))
    min_index = 0
    max_index = len(params)
    while True:
        ths_index = (min_index + max_index) // 2
        ths = params[ths_index]
        n = model.compute_params_count('structured', ths)

        if n == int(max * (1 - (target / 1000))):
            return ths
        elif n < int(max * (1 - (target / 1000))):
            max_index = ths_index
        else:
            min_index = ths_index

        if (max_index - min_index) <= 1:
            return ths


def get_structured_mask(model, target):
    ths = find_ths_by_dichotomy(target, model)
    masks = list()
    for name, p in model.named_parameters():
        if 'prunable' in name:
            if 'bias' in name:
                name_weight = name.replace('bias', 'weight')
                masks.append((model.state_dict()[name_weight].abs() >= ths).float())
            else:
                masks.append((p.abs() >= ths).float())
        else:
            masks.append((torch.ones(p.shape).to(p.device)).float())
    return masks


def get_unstructured_mask(model, target):
    parameters = torch.cat([i.abs().flatten() for n, i in model.named_parameters() if 'bn' not in n]).sort()[0]
    ths = parameters[int((target / 1000) * len(parameters))]
    return [(param.data.abs() >= ths).float() for param in model.parameters()]


class Regularization:
    def swd(self, model):
        mask = self.get_mask(model, self.target)
        total = 0
        for p, m in zip(model.parameters(), mask):
            total += 0.5 * self.a * torch.pow(p * (1 - m).float(), 2).sum()
        return total

    def baseline(self, model):
        return 0

    def liu2017(self, model):
        def smooth_l1(x):
            return torch.pow(x[x.abs() <= 1], 2).sum() + x[x.abs() > 1].sum()

        total = 0
        for n, p in model.named_modules():
            if 'prunable' in n:
                total += self.a * smooth_l1(p.weight)
        return total

    def __init__(self, a, target, args):
        self.a = a
        self.target = target
        if args.pruning_type == 'unstructured':
            self.get_mask = get_unstructured_mask
        elif args.pruning_type == 'structured':
            self.get_mask = get_structured_mask
        else:
            print('Wrong pruning type')
            raise ValueError

        if args.reg_type == 'swd':
            self.reg = self.swd
        elif args.reg_type == 'none':
            self.reg = self.baseline
        elif args.reg_type == 'liu2017':
            self.reg = self.liu2017
            if args.fix_a is None:
                print('Liu2017 reg must be performed with fix a')
                raise ValueError
        else:
            print('Wrong SWD type')
            raise ValueError

    def __call__(self, model):
        return self.reg(model)

    def set_a(self, a):
        self.a = a

    def get_target(self):
        return self.target


def get_mask_function(pruning_type):
    if pruning_type == 'unstructured':
        return get_unstructured_mask
    elif pruning_type == 'structured':
        return get_structured_mask
    else:
        print('Wrong pruning type')
        raise ValueError
