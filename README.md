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
Install PyTorch and torchvision 
```
$ conda install -c pytorch pytorch torchvision
$ pip install -r requirements.txt
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

To evaluate your own trained model on LA evaluation dataset:

```
python Main_training_script.py --track=logical --loss=CCE --is_eval --eval --model_path='path/to/your/best_model.pth' --eval_output='Eval_CM_scores_file.txt'
```

To evaluate a pre-trained RawNet2 on LA evaluation dataset:

```
python Main_training_script.py --track=logical --loss=CCE --is_eval --eval --model_path='/pre_trained_model/Best_model.pth' --eval_output='Eval_CM_scores_file_for_pre_trained_model.txt'
```

To compute scores on development dataset:

```
python Main_training_script.py --track=logical --loss=CCE --eval --model_path='S1_system_model.pth' --eval_output='Dev_CM_scores_file.txt'
```


Compute the min t-DCF and EER(%) on evaluation dataset

```
python evaluate_tDCF_asvspoof19_eval_LA.py 
``` 



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


