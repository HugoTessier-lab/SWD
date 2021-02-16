import torch
from utils.parse_arguments import parse_arguments
import sys
from utils.datasets import get_dataset
from utils.regularization_and_pruning import Regularization, get_mask_function
import math
import numpy as np
from utils.checkpoint import Checkpoint
import time

if torch.backends.cudnn.enabled:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

get_mask = None


def display_progress(batch_idx, accuracy, top5, loss, loader, train_or_test, batch_size):
    sys.stdout.write(f'\r{train_or_test} : ({batch_idx + 1}/{len(loader)}) '
                     f'-> top-1 : {round(accuracy / ((1 + batch_idx) * batch_size), 3)}'
                     f'     top-5 : {round(top5 / ((1 + batch_idx) * batch_size), 3)}'
                     f'     loss : {round(loss / ((1 + batch_idx) * batch_size), 3)}          ')


def apply_mask(model, masks):
    with torch.no_grad():
        for i, parameter in enumerate(model.parameters()):
            parameter.data = parameter.data * masks[i]


def test_model(dataset, model, args):
    model.eval()
    test_loader = dataset['test']
    accuracy = 0
    loss = 0
    top5 = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            if batch_idx != 0 and args.debug:
                break
            device = 'cuda' if not args.no_cuda else 'cpu'
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            accuracy += pred.eq(target.view_as(pred)).sum().item()
            top5 += accuracy_top5(output, target)
            loss += float(torch.nn.functional.cross_entropy(output, target))
            display_progress(batch_idx, accuracy, top5, loss, test_loader, 'Test', args.test_batch_size)
    return (accuracy / (len(test_loader) * args.test_batch_size),
            top5 / (len(test_loader) * args.test_batch_size),
            loss / (len(test_loader) * args.test_batch_size))


def compute_migration(before, after):
    ingoing = 0
    outgoing = 0
    for b, a in zip(before, after):
        ingoing += int(((b == 0) & (a == 1)).sum())
        outgoing += int(((b == 1) & (a == 0)).sum())
    return ingoing, outgoing


def l2_norm(model):
    norm = 0
    for p in model.parameters():
        norm += float(torch.pow(p, 2).sum())
    return norm


def get_a(batch_idx, current_epoch, max_epoch, dataset_length, args):
    if args.fix_a is not None:
        return args.fix_a
    else:
        max_batch = max_epoch * dataset_length
        current_batch = (current_epoch * dataset_length) + batch_idx
        exponent = math.log(args.a_max / args.a_min) / max_batch
        return args.a_min * math.exp(current_batch * exponent)


def accuracy_top5(output, target):
    result = 0
    for o, t in zip(output, target):
        result += int(t in torch.argsort(o, descending=True)[:5])
    return result


def train_model(checkpoint, args, epochs, dataset, masks=None, soft_pruning=False):
    while epochs[0] <= checkpoint.epoch < epochs[1]:
        if soft_pruning:
            apply_mask(checkpoint.model,
                       get_mask(checkpoint.model,
                                checkpoint.regularization.get_target() if checkpoint.regularization else args.target))
        if checkpoint.epoch == epochs[0]:
            acc, top5, test_loss = test_model(_dataset, checkpoint.model, args)
            checkpoint.save_results({'epoch': 'before', 'acc': acc, 'top5': top5, 'loss': test_loss,
                                     'norm': l2_norm(checkpoint.model),
                                     'pruned_param_count': checkpoint.model.compute_params_count(
                                         args.pruning_type),
                                     'pruned_flops_count': checkpoint.model.compute_flops_count()})
        print(f'\nEpoch {checkpoint.epoch + 1}/{epochs[1]}')
        train_loader = dataset['train']

        reg_mask_before = get_mask(checkpoint.model,
                                   checkpoint.regularization.get_target() if checkpoint.regularization else args.target)

        checkpoint.model.train()
        accuracy = 0
        global_loss = 0
        top5 = 0
        begin = None
        for batch_idx, (data, target) in enumerate(train_loader):
            if begin is None:
                print('Begin')
                begin = time.time()
            if batch_idx != 0 and args.debug:
                break
            if checkpoint.regularization is not None:
                checkpoint.regularization.set_a(get_a(batch_idx, checkpoint.epoch - epochs[0], epochs[1] - epochs[0],
                                                      len(dataset['train']), args))
            if masks:
                apply_mask(checkpoint.model, masks)

            device = 'cuda' if not args.no_cuda else 'cpu'
            data, target = data.to(device), target.to(device)
            checkpoint.optimizer.zero_grad()
            output = checkpoint.model(data)

            loss = torch.nn.functional.cross_entropy(output, target)
            if checkpoint.regularization:
                if args.wd == 0 and args.mu > 0:
                    loss += args.mu * checkpoint.regularization(checkpoint.model)
                else:
                    loss += args.wd * checkpoint.regularization(checkpoint.model)
            loss.backward()
            checkpoint.optimizer.step()

            pred = output.argmax(dim=1, keepdim=True)
            accuracy += pred.eq(target.view_as(pred)).sum().item()
            top5 += accuracy_top5(output, target)
            with torch.no_grad():
                global_loss += float(loss)
            display_progress(batch_idx, accuracy, top5, global_loss, train_loader, 'Train', args.batch_size)

        if masks:
            apply_mask(checkpoint.model, masks)

        duration = time.time() - begin

        reg_mask_after = get_mask(checkpoint.model,
                                  checkpoint.regularization.get_target() if checkpoint.regularization else args.target)
        ingoing, outgoing = compute_migration(reg_mask_before, reg_mask_after)
        last_a = get_a(len(dataset['train']) - 1, checkpoint.epoch - epochs[0], epochs[1] - epochs[0],
                       len(dataset['train']), args)

        sys.stderr.write('\n')
        acc, top5, test_loss = test_model(dataset, checkpoint.model, args)

        checkpoint.save_results({'epoch': checkpoint.epoch, 'acc': acc, 'top5': top5, 'loss': test_loss,
                                 'ingoing': ingoing, 'outgoing': outgoing, 'a': last_a,
                                 'norm': l2_norm(checkpoint.model),
                                 'pruned_param_count': checkpoint.model.compute_params_count(args.pruning_type),
                                 'pruned_flops_count': checkpoint.model.compute_flops_count(),
                                 'epoch_duration': duration})
        checkpoint.epoch += 1
        checkpoint.scheduler.step()
        checkpoint.save()


if __name__ == '__main__':
    arguments = parse_arguments()
    torch.manual_seed(arguments.seed)
    np.random.seed(arguments.seed)
    if arguments.fix_a is None and arguments.reg_type == "swd" and arguments.pruning_iterations != 1:
        print('Progressive a is not compatible with iterative pruning')
        raise ValueError
    if arguments.no_ft and arguments.pruning_iterations != 1:
        print("You can't specify a pruning_iteration value if there is no fine-tuning at all")
        raise ValueError
    get_mask = get_mask_function(arguments.pruning_type)
    _dataset = get_dataset(arguments)
    _targets = [int((n + 1) * (arguments.target / arguments.pruning_iterations)) for n in
                range(arguments.pruning_iterations)]

    # Train model
    print('Train model !')
    print(f'Regularization with t-{_targets[0]}')

    training_model = Checkpoint(arguments, 'training')
    training_model.regularization = Regularization(None, _targets[0], arguments)
    training_model.load()
    train_model(training_model, arguments, [0, arguments.epochs], _dataset, None, soft_pruning=arguments.soft_pruning)

    if arguments.lr_rewinding:
        training_model.rewind_lr()

    if arguments.no_ft:
        print('\nPruning model without fine tuning :')
        pruned_model = training_model.clone('pruned')
        pruned_model.load()
        mask = get_mask(pruned_model.model, arguments.target)
        apply_mask(pruned_model.model, mask)
        _acc, _top5, _test_loss = test_model(_dataset, pruned_model.model, arguments)
        pruned_model.save_results({'epoch': 'before', 'acc': _acc, 'top5': _top5, 'loss': _test_loss,
                                   'norm': l2_norm(pruned_model.model),
                                   'pruned_param_count': pruned_model.model.compute_params_count(
                                       arguments.pruning_type),
                                   'pruned_flops_count': pruned_model.model.compute_flops_count()})
        pruned_model.save()
        last_model = pruned_model
        last_epoch = arguments.epochs
    else:
        fine_tuned_model = training_model
        # Prune and fine-tune model
        for _i, _t in enumerate(_targets):
            print(f'\n\nPruning with target {_t}/1000 ({_i + 1}/{len(_targets)}) and fine-tuning model !')
            fine_tuned_model = fine_tuned_model.clone(f'fine_tuning({_i + 1}-{len(_targets)})')
            fine_tuned_model.load()
            mask = get_mask(fine_tuned_model.model, _t)

            if _i + 1 != len(_targets):
                regularization = Regularization(None, _targets[_i + 1], arguments)
                print(f'Regularization with t-{_targets[_i + 1]}')
            else:
                print('Final fine-tuning without regularization')
                regularization = None
            fine_tuned_model.regularization = regularization
            train_model(fine_tuned_model, arguments,
                        [arguments.epochs + (_i * arguments.ft_epochs),
                         arguments.epochs + ((_i + 1) * arguments.ft_epochs)],
                        _dataset, mask)
        last_model = fine_tuned_model
        last_epoch = arguments.epochs + (len(_targets) * arguments.ft_epochs)

    if arguments.additional_epochs != 0:
        print('\nAdditional fine-tuning epochs')
        print(last_model.epoch, last_epoch, last_epoch + arguments.additional_epochs)
        last_model = last_model.clone('last_epochs')
        last_model.load()
        last_model.regularization = None
        mask = get_mask(last_model.model, arguments.target)
        train_model(last_model, arguments,
                    [last_epoch,
                     last_epoch + arguments.additional_epochs],
                    _dataset, mask)

    print('\nDone')
