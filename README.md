End-to-end anti-spoofing with RawNet2
===============
This repository contains our implementation of the paper accepted to ICASSP 2021, "End-to-end anti-spoofing with RawNet2". This work demonstrates the effectivness of end-to-end approaches that utilise automatic feature learning to improve performance, both for seen spoofing attack types as well as for worst-case (A17) unseen attack.
[Paper link here](https://arxiv.org/abs/2011.01108)

## Installation
First, clone the repository locally, create and activate a conda environment, and install the requirements :
```
$ git clone https://github.com/TakHemlata/RawNet_anti_spoofing.git
$ conda create --name rawnet_anti_spoofing python=3.8.5
$ conda activate rawnet_anti_spoofing
$ conda install -c pytorch pytorch torchvision
$ pip install -r requirements.txt
```

## Experiments

### Dataset
Our model for the logical access (LA) track is trained on the LA train partition of the ASVspoof 2019 dataset, which can can be downloaded from [here](https://datashare.is.ed.ac.uk/handle/10283/3336).

### Training
To train the model run:
```
python main.py --track=logical --loss=CCE   --lr=0.0001 --batch_size=32
```

### Testing

To test your own model on the ASVspoof 2019 LA evaluation set:

```
python main.py --track=logical --loss=CCE --is_eval --eval --model_path='/path/to/your/your_best_model.pth' --eval_output='eval_CM_scores.txt'
```

We also provide a pre-trained model which follows a Mel-scale distribution of the sinc filters at the input layer. To use it you can run: 
```
python main.py --track=logical --loss=CCE --is_eval --eval --model_path='pre_trained_model/S1_system_model.pth' --eval_output='pre_trained_eval_CM_scores.txt'
```

If you would like to compute scores on the development set of ASVspoof 2019 simply run:

```
python main.py --track=logical --loss=CCE --eval --model_path='/path/to/your/best_model.pth' --eval_output='dev_CM_scores.txt'
```


To compute scores on development dataset:

```
python main.py --track=logical --loss=CCE --eval --model_path='S1_system_model.pth' --eval_output='Dev_CM_scores_file.txt'
```

Pre_trained weights are available in 'pre_trained_models/'  directory.
1. S1_system_model: fixed Mel-scaled sinc filters
2. S2_system_model: fixed inverse-mel-scaled sinc filters
3. S3_system_model: fixed linear-scaled sinc filters

Compute the min t-DCF and EER(%) on development and evaluation dataset
```
python tDCF_python/evaluate_tDCF_asvspoof19_eval_LA.py dev 'Dev_CM_scores_file.txt'
``` 

```
python tDCF_python/evaluate_tDCF_asvspoof19_eval_LA.py Eval 'eval_CM_scores.txt'
``` 

## Fusion
Fusion experiments performed using the "Support Vector Machine (SVM)"  based fusion approach. We trained SVM on development scores of all three RawNet2 systems with high-spectral resolution LFCC-GMM baseline system from [here](https://www.isca-speech.org/archive/Interspeech_2020/pdfs/1844.pdf) and tested on evaluation scores of all the systems.

SVM fusion script (matlab) is available in 'SVM_fusion/' directory 

## Contact
For any query regarding this repository, please contact:
- Hemlata Tak: tak[at]eurecom[dot]fr
## Citation
If you use this code in your research please use the following citation:
```bibtex
@INPROCEEDINGS{9414234,
  author={Tak, Hemlata and Patino, Jose and Todisco, Massimiliano and Nautsch, Andreas and Evans, Nicholas and Larcher, Anthony},
  booktitle={IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={End-to-End anti-spoofing with RawNet2}, 
  year={2021},
  pages={6369-6373}
}

```


