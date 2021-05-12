# ProxITR
Proximal Learning for Individualized Treatment Regimes Under Unmeasured Confounding

## Installation
### Requirement

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

- [data](./data/) contains data files or files to generate simulation data
- [src](./src/) source files:
    - [proxITR.py](./src/proxITR.py): main file of proximal ITR learning
    - [rkhs_scaler.py](./src/rkhs_scaler.py): estimators of ourcome bridge function h0 and treatment bridge function q0
    - [torchSVC.py](./src/torchSVC.py): optimizer of weighted binary support vector classification
- [notebooks](./notebooks): Usage examples

## Citation
Cite this repository by

```
@misc{qi2021proximal,
      title={Proximal Learning for Individualized Treatment Regimes Under Unmeasured Confounding}, 
      author={Zhengling Qi and Rui Miao and Xiaoke Zhang},
      year={2021},
      eprint={2105.01187},
      archivePrefix={arXiv},
      primaryClass={stat.ME}
}
```
