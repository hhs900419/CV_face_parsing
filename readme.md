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
- 60 comparison results are generated for better visualization of the model performance

### Notes
- model weight is too large, can't push to gihub
- [segmentation model pytorch](https://github.com/qubvel/segmentation_models.pytorch/tree/master) library might be useful for building different model architecture and applying pretrained weight.
- [this reference](https://github.com/hukenovs/easyportrait) given by TAs shows lots of performance results by using different model architecture
- [this library](https://github.com/open-mmlab/mmsegmentation) includes more architectures, but seems quite difficilt to use
- [10 samples from Unseen dataset](https://drive.google.com/drive/folders/1jbOs1aBDN3myl6WX47Qy8nUqp9svA8-j)
- [FaceSynthetics](https://github.com/microsoft/FaceSynthetics) dataset (optional)