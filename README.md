# SWD


## How to use :

Run (for example):

    python experiment.py --a 10 --target 950 --epochs 30 --ft_epochs 15 ... (list of parameters)
    
to launch the training. The process can be stopped and resumed later.

## Results :

Results are stored into two forms : 
1) *.chk, which store the trained model after the training and between each pruning step, as well as the results
2) *.txt which store the results for each epoch.

## Arguments :

* **-h, --help** : show this help message and exit
* **--checkpoint_path** : Where to save models (default: './checkpoint')
* **--dataset_path** : Where to get the dataset (default: './dataset')
* **--results_path** : Where to store the results summaries (default: './results')
* **--no_cuda** : Disables CUDA training
* **--distributed** : Distribute model across available GPUs (Warning : distribution should remain consistent before and after loading models)
* **--debug** : Limits each epoch to one backprop
* **--seed** : Manual seed for pytorch, so that all models start with the same initialisation (default: 0)
* **--dataset** : Which dataset to use between 'cifar10' and 'cifar100' (default: 'cifar10')
* **--model** : The model to load (default: 'resnet20')
* **--lr** : Learning rate (default: 0.1)
* **--lr_rewinding** : Rewinds LR for fine-tuning (default: False)
* **--wd** : Weight decay rate (default: 1e-4)
* **--mu** : If no weight decay but still need SWD, set mu to a value above 0 (default: -1)
* **--momentum** : Momentum of SGD (default: 0.9)
* **--batch_size** : Input batch size for training (default: 128)
* **--test_batch_size** : Input batch size for testing (default: 1000)
* **--epochs** : Number of epochs to train (default: 150)
* **--ft_epochs** : Number of epochs to fine-tune (default: 50)
* **--additional_epochs** : Number of epochs after the last fine-tuning (default: 0)
* **--pruning_iterations** : Amount of iterations into which subdivide the pruning process (default: 1)
* **--soft_pruning** : Prunes at the beginning of each epoch (default: False)
* **--pruning_type** : Type of pruning between "unstructured", "structured", which prunes batchnorms, and "structuredF", which also prunes corresponding filters (default: "unstructured")
* **--target** : Pruning rate (default: 900)
* **--no_ft** : Skips the fine-tuning and only prunes the model
* **--reg_type** : Type of regularization between "none", "swd" and "liu2017" (default: "none")
* **--a_min** : Parameter a of the SWD, minimum value and grows exponentially to a_max (default: 1e-1)
* **--a_max** : Parameter a of the SWD, maximum value (default: 1e5)
* **--fix_a** : Parameter a of the SWD, remains the same during the whole training process; if not None, overrides the other parameters (default: None)