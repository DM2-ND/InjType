# InjType

This is the implementation of paper [Injecting Entity Types into Entity-Guided Text Generation](https://arxiv.org/abs/2009.13401).

### Prerequisites
The recommended way to install the required packages is using pip and the provided `requirements.txt` file. Create the environment by running the following command: `pip install -r requirements.txt`

### Prepare Dataset
To prepare the dataset, run `python dataset/dataloader.py` in the top folder directory.

### Run the model
To run the model with Gigword Dataset <br>
(1) `NQG_HOME=/home_dir_replace_with_yours/InjType` <br>
(2) `bash $NQG_HOME/code/inj_gig.sh $NQG_HOME/dataset $NQG_HOME/code` <br>

To run the model with NYT Dataset <br>
(1) `NQG_HOME=/home_dir_replace_with_yours/InjType` <br>
(2) `bash $NQG_HOME/code/inj_nyt.sh $NQG_HOME/dataset $NQG_HOME/code`

### To evaluate the model
We use Texar Torch's BLEU score as one of the evaluation metrics.

### Citation

```
@article{dong2020inject,
  title={Injecting Entity Types into Entity-Guided Text Generation},
  author={Xiangyu Dong, Wenhao Yu, Chenguang Zhu, Meng Jiang},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2020}
}
```

### Aceknowledgements
This code was based in part on the source code of [NQG](https://github.com/magic282/NQG).



### Contact
If you have any question or suggestion, please send email to: \
Xiangyu Dong (```xdong2@nd.edu```) or Wenhao Yu (```wyu1@nd.edu```)