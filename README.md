# CSST--Reconstruction-Algorithm-of-Aperture-Diffraction-Imaging-Spectrometer

### Content Alerts
- More general experimental results under different exposure conditions are that the **CSST-9stg** can **improve close to 1.5 dB in PSNR** over the **Restormer** on the task of ADIS reconstruction.
- Notably, the algorithm can be used to solve the inverse problem of PSF engineering and RGB super-resolution.
- **This toolkit will be continually refined.**

### Acknowledgements
- We would like to express our gratitude to the author and contributors of MST-plus-plus and MST for their valuable work.
- CSST is an open-source project that leverages the functionality provided by external libraries called [MST-plus-plus](https://github.com/caiyuanhao1998/MST-plus-plus) and [MST](https://github.com/caiyuanhao1998/MST). 
- CSST modifies the underlying framework from [MST-plus-plus](https://github.com/caiyuanhao1998/MST-plus-plus) and [MST](https://github.com/caiyuanhao1998/MST) to perform a completely noval reconstruction task (ADIS) as well as PSF engineering. You can find the original MST-plus-plus repository [here](https://github.com/caiyuanhao1998/MST-plus-plus) and MST repository [here](https://github.com/caiyuanhao1998/MST). 




## 1. Result:
<img src="./figure/final-fig4s.png"  height=250 width=900>

<img src="./figure/table.png"  height=300 width=900>

**A more general result is that the CSST-9stg exhibits around 35dB in PSNR.**

&nbsp;



## 2. Create Environment:

- Python 3 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux))

- NVIDIA GPU + [CUDA](https://developer.nvidia.com/cuda-downloads)

- Python packages:

```shell
  pip install -r requirements.txt
```


&nbsp;


## 3. Prepare Dataset:

```shell
|--CSST
    |--Real
    	|-- test_code
    	|-- train_code
    |--simulation
    	|-- train_code
    |--tools
    |--datasets
        |--cave_1024_28
            |--scene1.mat
            |--scene2.mat
            ：  
            |--scene205.mat
        |--KAIST_CVPR2021_selected 
            |--1.mat
            |--2.mat
            ： 
            |--20.mat
        |--KAIST_CVPR2021_unselected 
            |--1.mat
            |--2.mat
            ： 
            |--10.mat

```

we use the CAVE dataset (cave_1024_28) and KAIST (KAIST_CVPR2021_selected) (10 secnes of KAIST_CVPR2021_unselected are croped as testset) as the simulation training set. Both the CAVE (CAVE_512_28) and KAIST (KAIST_CVPR2021) datasets are used as the real training set. 

We will soon make available the training and test sets used in the paper implementation.


&nbsp;


## 4. Simulation Experiement:

### 4.1　Training

```shell
python train.py --template CSST-3stg --outf ./exp/CSST-3stg/ --method CSST-3stg
python train.py --template CSST-5stg --outf ./exp/CSST-5stg/ --method CSST-5stg
python train.py --template CSST-7stg --outf ./exp/CSST-7stg/ --method CSST-7stg
python train.py --template CSST-9stg --outf ./exp/CSST-9stg/ --method CSST-9stg 
```

The training log, trained model, and reconstrcuted HSI will be available in `CSST/simulation/train_code/exp/` . 


### 4.2　Testing
By loading model weights in such a way that you can quickly implement model testing from the training code




### 4.3 Evaluating the Params and FLOPS of models

  We have provided a function `my_summary()` in `simulation/train_code/utils.py`, please use this function to evaluate the parameters and computational complexity of the models, especially the Transformers as 

```shell
from utils import my_summary
my_summary(CSST(), 256, 256, 28, 1)
```


&nbsp;


## 5. Real Experiement:

### 5.1　Training

```shell
python train.py --template CSST-3stg --outf ./exp/CSST-3stg/ --method CSST-3stg
python train.py --template CSST-5stg --outf ./exp/CSST-5stg/ --method CSST-5stg
python train.py --template CSST-7stg --outf ./exp/CSST-7stg/ --method CSST-7stg
python train.py --template CSST-9stg --outf ./exp/CSST-9stg/ --method CSST-9stg 
```

The training log, trained model, and reconstrcuted HSI will be available in `CSST/Real/train_code/exp/` . 


### 5.2　Testing

```shell
# first step
cd CSST/Real/test_code
python test_CSST_final.py
```

&nbsp;


## 6. Citation
If this repo helps you, please consider citing our works:


```shell
# ADIS+CSST
@inproceedings{lv2023aperture,
  title={Aperture Diffraction for Compact Snapshot Spectral Imaging},
  author={Lv, Tao and Ye, Hao and Yuan, Quan and Shi, Zhan and Wang, Yibo and Wang, Shuming and Cao, Xun},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={10574--10584},
  year={2023}
}
```









