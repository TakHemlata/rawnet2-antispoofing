

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
Our experiments are done in the logical access (LA) partition of the ASVspoof 2019 dataset, which can can be downloaded from [here](https://datashare.is.ed.ac.uk/handle/10283/3336).

### Training
To train the model run:
```
python main.py --track=logical --loss=CCE   --lr=0.0001 --batch_size=32
```

### Testing

To evaluate your own model on LA evaluation dataset:

```
python main.py --track=logical --loss=CCE --is_eval --eval --model_path='/path/to/your/best_model.pth' --eval_output='eval_CM_scores_file.txt'
```

We also provide a pre-trained model which follows a Mel-scale distribution of the sinc filters at the input layer. To use it you can run: 
```
python main.py --track=logical --loss=CCE --is_eval --eval --model_path='pre_trained_model/best_model.pth' --eval_output='RawNet2_LA_eval_CM_scores.txt'
```

If you would like to compute scores on development dataset simply run:

```
python main.py --track=logical --loss=CCE --eval --model_path='/path/to/your/best_model.pth' --eval_output='dev_CM_scores_file.txt'
```
Compute the min t-DCF and EER(%) on development dataset
```
python tDCF_python/evaluate_tDCF_asvspoof19_eval_LA.py  dev  'dev_CM_scores_file.txt'
``` 

Compute the min t-DCF and EER(%) on evaluation dataset
```
python tDCF_python/evaluate_tDCF_asvspoof19_eval_LA.py  Eval  'eval_CM_scores_file.txt'
``` 
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
