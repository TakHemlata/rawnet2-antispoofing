End-to-end anti-spoofing with RawNet2
===============
This repository contains our implementation of the paper accepted to ICASSP 2021, "End-to-end anti-spoofing with RawNet2". This work demonstrates the effectivness of end-to-end approaches that utilise automatic feature learning to improve performance, both for seen spoofing attack types as well as for worst-case (A17) unseen attack.
[Paper link here](https://arxiv.org/abs/2011.01108)

## Usage
First, clone the repository locally:
```
$ git clone https://github.com/TakHemlata/RawNet_anti_spoofing.git
```
Then, to create an environment:

```
$ conda create --name rawnet_anti_spoofing python=3.8.5
$ conda activate rawnet_anti_spoofing
```
Install PyTorch 1.5.1 and torchvision 0.6.1
```
$ conda install -c pytorch pytorch torchvision
$ pip install PyYAML == 5.4.1
```


## Dataset

The ASVSpoof2019  dataset can be downloaded from the following link:

[ASVSpoof2019 dataset](https://datashare.is.ed.ac.uk/handle/10283/3336)
### Training models
To train RawNet2 run:
```
python Main_training_script.py --track=logical --loss=CCE   --lr=0.0001 --batch_size=32
```

## Evaluation
To evaluate a pre-trained RawNet2 on asvspoof evaluation dataset:

```
python Main_training_script.py --track=logical --loss=CCE --is_eval --eval --model_path='S3_system_model.pth' --eval_output='Eval_scores_file.txt'
```
We provide RawNet2 models pretrained on ASVspoof 2019 logical access (LA) database. 
| Systems | EER (%) | min t-DCF | 
| --- | --- | --- | 
| L | 3.50 |0.0904  | 
| S1 | 4.52 | 0.1059 | 
| S2 | 4.99 | 0.1204 |
| S3 |4.62  |0.1125  | 
L: High-spectral-resolution LFCC baseline (https://www.isca-speech.org/archive/Interspeech_2020/pdfs/1844.pdf)


Pre_trained weights are avaiable in "pre_trained_models/"  directory.
1. S1_system_model: fixed Mel-scaled sinc filters
2. S2_system_model: fixed inverse-mel-scaled sinc filters
3. S3_system_model: fixed linear-scaled sinc filters

Compute the min t-DCF and EER(%) on evaluation dataset

```
python evaluate_tDCF_asvspoof19_eval_LA.py 
``` 
## Fusion
Performed fusion experoiments using the "support vector machine (SVM)"  based fusion approach. We trained SVM on development scores of all RawNet2 systaems along with LFCC-GMM baseline and test on evaluation scores.

SVM fusion script (matlab) is avaiable in "SVM_fusion/" directory with development and evaluation scores and coressponding labels of all countermeasures system in "S_dev.mat" and "S_eval.mat" respectively.

## Contact
For any query, please contact:
- Hemlata Tak: tak[at]eurecom[dot]fr
## Citation
If you use this code for a paper please cite:
```bibtex
@article{tak2020end,
  title={End-to-end anti-spoofing with RawNet2},
  author={Tak, Hemlata and Patino, Jose and Todisco, Massimiliano and Nautsch, Andreas and Evans, Nicholas and Larcher, Anthony},
  journal={arXiv preprint arXiv:2011.01108},
  year={2020}
}
```


