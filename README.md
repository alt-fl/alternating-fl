# Alt-FL

Todo...

## Example experiments using Alt-FL

Suppose we have a FL scenario where there are 3 clients, and all clients are training for 5 local epochs each round, then we can run the following command.

- With differential privacy enabled (DP epsilon=5):
```bash
python main.py --K 3 --C 1 --r 400 --E 5 --model LeNet5 --dataset CIFAR10 --exp_repeat 10 --strategy FedAvg --epsilon 0 --rho_syn 0 --rho_tot 1 --lr 0.005 --optimizer adam --dims_feature 84 --weight_decay 1 --num_classes 10 --syn_balance self --init_syn_rounds 0 --use_dp --dp_epsilon 5 --window_size 12 --patience 10 --early_stop_delta 0.001 --output output.pt --save_every 20 --device cpu
```

- With selective homomorphic encryption enabled (encryption ratio=0.2), note that the argument for encryption ratio is called `--epsilon` which can be confused with the privacy budget parameter in DP (e.g., `--dp_epsilon`, but we will change it later):
```bash
python main.py --K 3 --C 1 --r 400 --E 5 --model LeNet5 --dataset CIFAR10 --exp_repeat 10 --strategy FedAvg --epsilon 0.2 --rho_syn 0 --rho_tot 1 --lr 0.0005 --optimizer adam --dims_feature 84 --weight_decay 1 --num_classes 10 --syn_balance self --init_syn_rounds 0 --window_size 12 --patience 10 --early_stop_delta 0.001 --output output.pt --save_every 20 --device cpu
```

- Without any privacy defenses, but with interleaving ratio 0.5:
```bash
python main.py --K 3 --C 1 --r 400 --E 5 --model LeNet5 --dataset CIFAR10 --exp_repeat 10 --strategy FedAvg --epsilon 0 --rho_syn 1 --rho_tot 2 --lr 0.0005 --optimizer adam --dims_feature 84 --weight_decay 1 --num_classes 10 --syn_balance self --init_syn_rounds 0 --window_size 12 --patience 10 --early_stop_delta 0.001 --output output.pt --save_every 20 --device cpu
```

Run `python main.py --help` for an explanation of what each argument does.

## Build Alt-FL using Apptainer

An example specification file for Alt-FL is provided in `alt-fl.def`, use the
following command to build an `.sif` image:
```bash
apptainer build alt-fl.sif alt-fl.def 

```

Then run it with this command:
```bash
apptainer run --bind <PATH_TO_ALT_FL>:/app <PATH_TO_IMAGE> [ALT_FL_ARGUMENTS]
```

