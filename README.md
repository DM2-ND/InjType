# InjType

This is the implementation of **EMNLP 2021** paper **[Injecting Entity Types into Entity-Guided Text Generation](https://arxiv.org/abs/2009.13401)**.

In this work, we aim to enhance the role of entity in NLG model to help generate sequence accurately. Specifically, we develop a novel NLG model to produce a target sequence based on a given list of entities. Our model has a multi-step decoder that injects the entity types into the process of entity mention generation. Experiments on two public news datasets (i.e., Gigawords and NYT) demonstrate type injection performs better than existing type embedding concatenation baselines. 


### Prerequisites

**1. Using pip:** The recommended way to install the required packages is using pip and the provided `requirements.txt` file. Create the environment by running the following command: `pip install -r requirements.txt`

**2. Using docker:** The second way is to pull our docker image on the [dockerhub](https://hub.docker.com/). Download the docker image by using the following command: `docker pull wenhaoyu97/injtype:gen1.0`

### Prepare Dataset
To prepare the dataset, run `python dataset/dataloader.py` in the top folder directory.

### Run the model
To run the model with Gigawords Dataset <br>
(1) `NQG_HOME=/home_dir_replace_with_yours/InjType` <br>
(2) `bash $NQG_HOME/scripts/inj_gig.sh $NQG_HOME` <br>

To run the model with NYT Dataset <br>
(1) `NQG_HOME=/home_dir_replace_with_yours/InjType` <br>
(2) `bash $NQG_HOME/scripts/inj_nyt.sh $NQG_HOME`

### To evaluate the model
We use [Texar-torch BLEU](https://github.com/asyml/texar-pytorch) score and [PyPI ROUGE](https://pypi.org/project/rouge/) to evaluate model performance.

### Citation
```
@inproceedings{dong2021injecting,
  title={Injecting Entity Types into Entity-Guided Text Generation},
  author={Dong, Xiangyu and Yu, Wenhao and Zhu, Chenguang and Jiang, Meng},
  booktitle={Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  year={2021}
}
```

### Aceknowledgements
This code was based in part on the source code of [NQG](https://github.com/magic282/NQG).

### Contact
If you have any question or suggestion, please send email to: \
Xiangyu Dong (```xdong2ps@gmail.com```) or Wenhao Yu (```wyu1@nd.edu```)
