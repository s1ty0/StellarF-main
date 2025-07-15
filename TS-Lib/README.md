Before we start our experiment, please follow the instructions below to set up the environment.

```
conda create -n tu_m python=3.8
conda activate tu_m
pip install requirements.txt
```

Now, run as below

```
sh ./scipts/classification/MICN.py
```

Among them, there are 6 models. Simply execute the corresponding scripts for each one. 
Note: If you need to reproduce the experiments after adding new innovative points, please write the corresponding script and switch to the following instructions (choose one from the four):

```
  --root_path ./dataset_k \
  --root_path ./dataset_k12 \
  --root_path ./dataset_t \
  --root_path ./dataset_t12 \
```

Wish you a smooth reproduction!

