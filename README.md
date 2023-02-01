# ProxITR
Proximal Learning for Individualized Treatment Regimes Under Unmeasured Confounding

## Requirements

```
python >= 3.8
numpy >= 1.20
scipy >= 1.6.2
pandas >= 1.2.3
scikit-learn >= 0.24.1
pytorch >= 1.8.1
```

### Conda Installation

```
conda install pandas scikit-learn numpy scipy pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c conda-forge
```

## Content

- [notebooks](./notebooks): Usage examples for all simulation settings in the paper, in each example:
    - `samp_size`: sample size is 2000 (users can try other settings)
    - `qtl`: the quantile for dPESS selection, default is 0.4 (users can try other settings)
- [data](./data/) contains file to generate simulated data
- [src](./src/) source files:
    - [proxITR.py](./src/proxITR.py): main file of proximal ITR learning
    - [rkhs_scaler.py](./src/rkhs_scaler.py): estimators of ourcome bridge function h0 and treatment bridge function q0
    - [torchSVC.py](./src/torchSVC.py): optimizer of weighted binary support vector classification

## Citation

```
@article{qi2022proximal,
  title={Proximal learning for individualized treatment regimes under unmeasured confounding},
  author={Qi, Zhengling and Miao, Rui and Zhang, Xiaoke},
  journal={Journal of the American Statistical Association},
  number={just-accepted},
  pages={1--33},
  year={2022},
  publisher={Taylor \& Francis}
}

```
