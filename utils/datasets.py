import torch
from torchvision import datasets, transforms
import os


def load_mnist(args):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(args.dataset_path, train=True, download=True, transform=transform),
        batch_size=args.batch_size, shuffle=True
    )

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(args.dataset_path, train=False, download=True, transform=transform),
        batch_size=args.test_batch_size, shuffle=True
    )

    return {'train': train_loader, 'test': test_loader}


def load_cifar10(args):
    list_trans = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(),
                  transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]

    transform_train = transforms.Compose(list_trans)

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(args.dataset_path, train=True, download=True, transform=transform_train),
        batch_size=args.batch_size, shuffle=True, num_workers=4)

    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(args.dataset_path, train=False, download=True, transform=transform_test),
        batch_size=args.test_batch_size, shuffle=True, num_workers=4)

    return {'train': train_loader, 'test': test_loader}


def load_cifar100(args):
    list_trans = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(),
                  transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]

    transform_train = transforms.Compose(list_trans)

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100(args.dataset_path, train=True, download=True, transform=transform_train),
        batch_size=args.batch_size, shuffle=True, num_workers=4)

    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100(args.dataset_path, train=False, download=True, transform=transform_test),
        batch_size=args.test_batch_size, shuffle=True, num_workers=4)

    return {'train': train_loader, 'test': test_loader}


def load_imagenet(args):
    # train_dir = os.path.join(args.dataset_path, 'train')
    test_dir = os.path.join(args.dataset_path, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    # train_dataset = datasets.ImageFolder(
    #     train_dir,
    #     transforms.Compose([
    #         transforms.RandomResizedCrop(224),
    #         transforms.RandomHorizontalFlip(),
    #         transforms.ToTensor(),
    #         normalize,
    #     ]))

    train_dataset = datasets.ImageNet(args.dataset_path, split='train', transform=transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ]))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=True, sampler=None)

    test_dataset = datasets.ImageNet(args.dataset_path, split='val', transform=transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ]))

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.test_batch_size, shuffle=False,
        num_workers=4, pin_memory=True)

    return {'train': train_loader, 'test': test_loader}


def get_dataset(args):
    if args.dataset == "cifar10":
        return load_cifar10(args)

    elif args.dataset == "cifar100":
        return load_cifar100(args)

    elif args.dataset == "imagenet":
        return load_imagenet(args)

    elif args.dataset == "mnist":
        return load_mnist(args)

    else:
        raise Exception(f"Dataset '{args.dataset}' is no recognized dataset. Could not load any data.")
