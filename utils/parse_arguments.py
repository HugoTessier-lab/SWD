import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description='SWD')

    parser.add_argument('--checkpoint_path', type=str, default="./checkpoint",
                        help="Where to save models  (default: './checkpoint')")

    parser.add_argument('--dataset_path', type=str, default="./dataset",
                        help="Where to get the dataset (default: './dataset')")

    parser.add_argument('--results_path', type=str, default="./results",
                        help="Where to store the results summaries (default: './results')")

    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='Disables CUDA training')

    parser.add_argument("--debug", action="store_true",
                        help="Limits each epoch to one backprop")

    parser.add_argument("--seed", type=float, default=0,
                        help="Manual seed for pytorch, so that all models start with the same initialisation "
                             "(default: 0)")

    parser.add_argument('--dataset', type=str, default="cifar10",
                        help="Which dataset to use between 'cifar10', 'cifar100' or 'imagenet' (default: 'cifar10')")

    parser.add_argument('--model', type=str, default="resnet18",
                        help="The model to load (default: 'resnet18')")

    parser.add_argument('--lr', type=float, default=0.1,
                        help='Learning rate (default: 0.1)')

    parser.add_argument('--wd', default="5e-4", type=float,
                        help='Weight decay rate (default: 5e-4)')

    parser.add_argument('--mu', default="-1", type=float,
                        help='If no weight decay but still need SWD, set mu to a value above 0 (default: -1)')

    parser.add_argument('--momentum', default="0.9", type=float,
                        help='Momentum of SGD (default: 0.9)')

    parser.add_argument('--batch_size', type=int, default=128,
                        help='Input batch size for training (default: 128)')

    parser.add_argument('--test_batch_size', type=int, default=1000,
                        help='Input batch size for testing (default: 1000)')

    parser.add_argument('--epochs', type=int, default=300,
                        help='Number of epochs to train  (default: 300)')

    parser.add_argument('--ft_epochs', type=int, default=50,
                        help='Number of epochs to fine-tune (default: 50)')

    parser.add_argument('--additional_epochs', type=int, default=0,
                        help='Number of epochs after the last fine-tuning (default: 0)')

    parser.add_argument('--pruning_iterations', type=int, default=1,
                        help='Amount of iterations into which subdivide the pruning process (default: 1)')

    parser.add_argument('--pruning_type', type=str, default='unstructured',
                        help='Type of pruning between "unstructured", '
                             '"structured" (default: "unstructured")')

    parser.add_argument('--target', default="900", type=float, help="Pruning rate  (default: 900)")

    parser.add_argument("--no_ft", action="store_true",
                        help="Skips the fine-tuning and only prunes the model")

    parser.add_argument('--reg_type', type=str, default='none',
                        help='Type of regularization between "none", "swd" and "liu2017" (default: "none")')

    parser.add_argument('--a_min', default=1e-1, type=float,
                        help='Parameter a of the SWD, minimum value and grows exponentially to a_max (default: 1e-1)')

    parser.add_argument('--a_max', default=1e5, type=float,
                        help='Parameter a of the SWD, maximum value (default: 1e5)')

    parser.add_argument('--fix_a', default=None, type=float,
                        help='Parameter a of the SWD, remains the same during the whole training process; '
                             'if not None, overrides the other parameters (default: None)')

    return parser.parse_args()
