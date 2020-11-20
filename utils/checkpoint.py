import utils.resnet as resnet
import utils.mobilenet as mobilenet
import utils.lenet5 as lenet5
import torch
import os
import copy


class Checkpoint:
    def __init__(self, args, name):
        self.args = args
        self.results = list()
        self.epoch = None

        self.model = None
        self.optimizer = None
        self.scheduler = None

        self.regularization = None

        self.name = name
        self.create()
        self.load()

    def training_specs(self):
        indicate_target = not ('training' in self.name and not self.args.reg_type == 'swd')
        return (f"(model-{self.args.model}_dataset-{self.args.dataset}_wd-{self.args.wd}" +
                (f"_pruningType-{self.args.pruning_type}_pruningTarget-{self.args.target}" if indicate_target else "")
                + ")_")

    def reg_specs(self):
        if self.args.reg_type == 'none':
            return f'(reg_type-None)_'
        elif self.args.reg_type == 'liu2017':
            return f'(reg_type-{self.args.reg_type}_lambda-{self.args.fix_a})_'
        else:
            if self.args.fix_a is not None:
                return f'(reg_type-{self.args.reg_type}_aFix-{self.args.fix_a}_' \
                       f'target-{self.regularization.target if self.regularization is not None else "none"})_'
            else:
                return f'(reg_type-{self.args.reg_type}_a-({self.args.a_min}-{self.args.a_max})_' \
                       f'target-{self.regularization.target if self.regularization is not None else "none"})_'

    def get_file_name(self, name):
        name = self.training_specs() + self.reg_specs() + name
        if self.args.debug:
            name += '_DEBUG'
        return name

    def create(self):
        if self.args.model not in resnet.__all__ and self.args.model not in mobilenet.__all__ and self.args.model != 'lenet5':
            print('Invalid model')
            raise ValueError
        if self.args.model in resnet.__all__:
            if self.args.dataset == 'cifar10':
                num_class = 10
            elif self.args.dataset == 'cifar100':
                num_class = 100
            elif self.args.dataset == 'imagenet':
                num_class = 1000
            elif self.args.dataset == 'mnist':
                num_class = 10
            else:
                print("Wrong dataset specified")
                raise ValueError
            self.model = resnet.resnet_model(self.args.model, num_class)
        elif self.args.model == 'mobilenet':
            self.model = mobilenet.mobilenet_v2()
        else:
            self.model = lenet5.LeNet5()
        if not self.args.no_cuda:
            self.model = self.model.cuda()
        self.optimizer = torch.optim.SGD(self.model.parameters(),
                                         lr=self.args.lr, momentum=self.args.momentum, weight_decay=self.args.wd)
        if self.args.dataset == 'mnist':
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer,
                                                                  milestones=[self.args.epochs * 2],
                                                                  gamma=0.1)
        else:
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer,
                                                                  milestones=[self.args.epochs // 3,
                                                                              2 * self.args.epochs // 3],
                                                                  gamma=0.1)
        self.epoch = 0

    def save(self):
        path = os.path.join(self.args.checkpoint_path, self.get_file_name(self.name) + '.chk')
        checkpoint = {'args': self.args, 'results': self.results, 'epoch': self.epoch,
                      'model_state_dict': self.model.state_dict(),
                      'optimizer_state_dict': self.optimizer.state_dict(),
                      'scheduler': self.scheduler,
                      'regularization': self.regularization}
        torch.save(checkpoint, path)
        path = os.path.join(self.args.results_path, self.get_file_name(self.name) + '.txt')
        with open(path, 'w') as f:
            f.truncate(0)
            f.write(str(self.args))
            f.write('\n')
            for r in self.results:
                f.write(str(r))
                f.write('\n')

    def load(self):
        path = os.path.join(self.args.checkpoint_path, self.get_file_name(self.name) + '.chk')
        if os.path.isfile(path):
            if self.model is None or self.optimizer is None or self.scheduler is None:
                self.create()
            checkpoint = torch.load(path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler = checkpoint['scheduler']
            self.results = checkpoint['results']
            self.epoch = checkpoint['epoch']
            self.regularization = checkpoint['regularization']

    def save_results(self, r):
        self.results.append(r)

    def clone(self, new_name):
        clone = copy.deepcopy(self)
        clone.name = new_name
        return clone
