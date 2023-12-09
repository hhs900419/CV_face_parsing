## Training
1. Prepare training data :
    -- download [CelebAMask-HQ dataset](https://github.com/switchablenorms/CelebAMask-HQ)

2. Preprocessing
(to stack all the mask images of a person into one)
```Shell
python prepropess_data.py
```

2. Train
```Shell
python train.py
```

- Current model (Unet)
- loss function: $Loss = cross entropy + dice loss$
- evaluation metric: mean IoU 
- hyperparamters and global variables are in `configs.py`
- beaware of the arguments in the scheduler, unexpected result are produced if the scheduler is initialized unappropriately 
- a jupyter notebook is also created for visualization convenience.

### Inference
```Shell
python test.py
```