Robust Network Enhancement from Flawed Networks
===============================================================================

About
-----

This repository is an implementation of the following paper:

Xu, J; Yang, Y; Wang, C; Liu, Z; Zhang, J; Chen, L and Lu, J, Robust Network Enhancement from Flawed Networks, IEEE Transactions
on Knowledge and Data Engineering, 2020. 
[<a href="https://galina0217.github.io/works/enet/tkde2020_robust_xu.pdf">PDF</a>]

```
@article{xu2020robust,
  title={Robust Network Enhancement from Flawed Networks},
  author={Xu, Jiarong and Yang, Yang and Wang, Chunping and Liu, Zongtao and Zhang, Jing and Chen, Lei and Lu, Jiangang},
  journal={IEEE Transactions on Knowledge and Data Engineering},
  year={2020},
  publisher={IEEE}
}
```

Dependencies
-----

The script has been tested running under Python 2.7.12, with the following packages installed (along with their dependencies):

- `torch==1.1.0`
- `networkx==2.0`
- `sklearn==0.19.1`
- `numpy==1.11.0`
- `scipy==1.1.0`
- `gensim==3.6.0`
- `tqdm==4.19.4`

Some Python module dependencies are listed in `requirements.txt`, which can be easily installed with pip:

```
pip install -r requirements.txt
```

In addition, CUDA 8.0 has been used in our project. Although not all dependencies are mentioned in the installation instruction links above, you can find most of the libraries in the package repository of a regular Linux distribution.


Input Format
-----
An example data format is given in ```Enet/data``` where dataset is in ```mat``` format and ```score``` is some heuristic scores calculated in advance. Besides, ```Enet/embedding```(OPTIONAL) provides additional node embedding results by Node2vec (https://github.com/aditya-grover/node2vec).

When using your own dataset, you must provide:

* an N by N adjacency matrix (N is the number of nodes).
* an N by D feature matrix (D is the dimension of node features).
* an N by C one-hot label matrix (C is the number of classes).
* an N by K heuristic score matrix (K is the number of heuristic scores).

How to run E-net
-----
The help information of the main script ```Enet/Main.py``` is listed as follows:

    python Main.py -h
    
    usage: Main.py [-h][--data-name] [--save-name] [--max-train-num] [--no-cuda] [--missing-ratio] 
    [--split-ratio] [--neg-pos-ratio] [--use-attribute] [--use-embedding] [--embedding-size] 
    [--lazy-subgraph] [--max-nodes-per-hop] [--num-walks] [--multi-subgraph] [--reg-smooth] 
    [--smooth-coef] [--trainable-noise] [--early-stop] [--early-stop-patience] [--learning-rate] 
    
    optional arguments:
      -h, --help                show this help message and exit
      --data-name               str, select the dataset. 
      --save-name               str, the name of saved model. 
      --max-train-num           int, the maximum number of training links, default 100000.
      --no-cuda                 bool, whether to disables CUDA training, default False.
      --seed                    int, set the random seed, default 1.
      --test-ratio              float, the ratio of test links, default 0.1.
      --missing-ratio           float, the ratio of missing links, default 0.1.
      --split-ratio             str, the split rate of train, val and test links, default 0.8,0.1,0.1.
      --neg-pos-ratio           float, the ratio of negative/positive links, default 5.
      --use-attribute           bool, whether to utilize node attribute, default True. 
      --use-embedding           bool, whether to utilize the information from node2vec node embeddings, default False.
      --embedding-size          int, the embedding size of node2vec, default 128.
      --lazy-subgraph           bool, whether to use lazy subgraph extraction, default True.
      --max-nodes-per-hop       int, the upper bound the number of nodes per hop when performing Lazy Subgraph Extraction, default 20.
      --num-walks               int, thenumber of walks for each node when performing Lazy Subgraph Extraction, default 5.
      --multi-subgraph          int, the number of subgraphs to extract for each queried nodes, default 3.
      --reg-smooth              bool, whether to use auxiliary denoising regularization, default False.
      --smooth-coef             float, the coefficient of auxiliary denoising regularization, default 1e-4.
      --trainable-noise         bool, whether to let the Noisy link detection layer trainable, default False.
      --early-stop              bool, whether to use early stopping, default True. 
      --early-stop-patience     int, the patience for early stop, default 7. 
      --learning-rate           float, the learning rate, default 1e-4. 

To reproduce the results that reported in the paper, you can run the following command.

- To train the variant E-Net (n)
```bash
python Main.py --data-name cora --use-embedding --num-walks 5 --learning-rate 1e-4 
 --noise-hidden-dim 500 --use-sig --use-soft --reg-smooth --smooth-coef 1e-4 --trainable-noise 
```

- To train the variant E-Net (fix)
```bash
python Main.py --data-name cora --num-walks 5 --learning-rate 1e-4
--noise-hidden-dim 500 --use-sig --use-soft --reg-smooth --smooth-coef 1e-4
```

- To train the variant E-Net (s-)
```bash
python Main.py --data-name cora --num-walks 5 --learning-rate 1e-4 
--noise-hidden-dim 500 --use-sig --use-soft --reg-smooth --smooth-coef 1e-4 --trainable-noise 
--smooth-coef 0
```

- To train the variant E-Net
```bash
python Main.py --data-name cora --num-walks 5 --learning-rate 1e-4 
--noise-hidden-dim 500 --use-sig --use-soft --reg-smooth --smooth-coef 1e-4 --trainable-noise 
```
