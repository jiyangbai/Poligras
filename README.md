# Poligras


An implementation of the "POLIGRAS: Policy-based Graph Summarization".



<!-- A reference Tensorflow implementation is accessible [[here]](https://github.com/yunshengb/SimGNN) and another implementation is [[here]](https://github.com/NightlyJourney/SimGNN). -->

<!-- ### Requirements
The codebase is implemented in Python 3.5.2. package versions used for development are just below.
```
networkx          2.4
tqdm              4.28.1
numpy             1.15.4
pandas            0.23.4
texttable         1.5.0
scipy             1.1.0
argparse          1.1.0
torch             1.1.0
torch-scatter     1.4.0
torch-sparse      0.4.3
torch-cluster     1.4.5
torch-geometric   1.3.2
torchvision       0.3.0
scikit-learn      0.20.0
``` -->
### Datasets
<p align="justify">
All datasets can be accessed <a href="https://drive.google.com/drive/folders/1v0CGwxQq2sgmraaWWD9nF9OFgwdb44Nv?usp=sharing" target="_blank">[here]</a>. We have uploaded the astro-ph and cnr-200 (in .zip file) into this github file. Before running code on specific dataset, please make sure to create file directory with same name as dataset at first, then download and unzip dataset files from the given link and put them into the created file directory. For example, if running on the in-2004 dataset, first create directory <code>./dataset/in-2004/</code>, then download and unzip the in-2004 dataset (including graph file and node features file) into the directory (i.e., having <code>./dataset/in-2004/in-2004</code> and <code>./dataset/in-2004/in-2004_feat</code>).</p>

<!-- Every JSON file has the following key-value structure:

```javascript
{"graph_1": [[0, 1], [1, 2], [2, 3], [3, 4]],
 "graph_2":  [[0, 1], [1, 2], [1, 3], [3, 4], [2, 4]],
 "labels_1": [2, 2, 2, 2],
 "labels_2": [2, 3, 2, 2, 2],
 "ged": 1}
```
<p align="justify">
The **graph_1** and **graph_2** keys have edge list values which descibe the connectivity structure. Similarly, the **labels_1**  and **labels_2** keys have labels for each node which are stored as list - positions in the list correspond to node identifiers. The **ged** key has an integer value which is the raw graph edit distance for the pair of graphs.</p>

### Options
<p align="justify">
Training a SimGNN model is handled by the `src/main.py` script which provides the following command line arguments.</p>

#### Input and output options
```
  --training-graphs   STR    Training graphs folder.      Default is `dataset/train/`.
  --testing-graphs    STR    Testing graphs folder.       Default is `dataset/test/`.
```
#### Model options
```
  --filters-1             INT         Number of filter in 1st GCN layer.       Default is 128.
  --filters-2             INT         Number of filter in 2nd GCN layer.       Default is 64. 
  --filters-3             INT         Number of filter in 3rd GCN layer.       Default is 32.
  --tensor-neurons        INT         Neurons in tensor network layer.         Default is 16.
  --bottle-neck-neurons   INT         Bottle neck layer neurons.               Default is 16.
  --bins                  INT         Number of histogram bins.                Default is 16.
  --batch-size            INT         Number of pairs processed per batch.     Default is 128. 
  --epochs                INT         Number of SimGNN training epochs.        Default is 5.
  --dropout               FLOAT       Dropout rate.                            Default is 0.5.
  --learning-rate         FLOAT       Learning rate.                           Default is 0.001.
  --weight-decay          FLOAT       Weight decay.                            Default is 10^-5.
  --histogram             BOOL        Include histogram features.              Default is False.
``` -->

### Dependencies

<p align="justify">
 Install the following tools and packages:
<ul dir="auto">
<li>
    <code>python3</code>
    : Assume 
    <code>python3</code>
     by default (use 
    <code>pip3</code>
     to install packages).
</li>
<li>
    <code>numpy</code>
</li>
<li>
    <code>random</code>
</li>
 <li>
    <code>torch</code>
</li>
 <li>
    <code>networkx</code>
</li>
<li>
    <code>copy</code>
</li>
<li>
    <code>argparse</code>
</li>
 <li>
    <code>pickle</code>
</li>
 <li>
    <code>glob</code>
 </li>
</ul>
</p>


### To run the code
<p align="justify">

The <code>run.py</code> is to set up all hyperparameters; the <code>model.py</code> includes model details; the <code>node_feature_generation.py</code> is to generate node features for the given graph.
 
The following commands train and execute Poligras model on specific dataset.</p>

```
python3 src/run.py --dataset dataset_name
```

After the running, the total rewards will be printed and users can store the generated graph summaries.
<!-- Training a SimGNN model for a 100 epochs with a batch size of 512.
```
python src/main.py --epochs 100 --batch-size 512
```
Training a SimGNN with histogram features.
```
python src/main.py --histogram
```
Training a SimGNN with histogram features and a large bin number.
```
python src/main.py --histogram --bins 32
```
Increasing the learning rate and the dropout.
```
python src/main.py --learning-rate 0.01 --dropout 0.9
``` -->
