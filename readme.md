# Balor


This is the codebase for my ICCAD'24 publication "Balor: HLS Source Code Evaluator Based on Custom Graphs and Hierarchical GNNs". Balor was designed to facilitate design space exploration for high-level synthesis, using graph neural networks to estimate the Quality of Results quickly and accurately.

Winner of stage 1 of [ML Contest for Chip Design with HLS](https://www.kaggle.com/competitions/machine-learning-contest-for-high-level-synthesis/leaderboard), hosted by UCLA Vast lab, UCLA DM lab, and AMD.

Our evaluate functions in run.py generate detailed reports on estimation error across the 6 metrics: LUTs, FFs, Latency, Clock Period, DSPs and BRAMs. The pre-generated sample report "db4hls_pretrained_report.pdf" is available with error statistics, scatter plots, error histograms and numerical cdfs.


### Using Balor

The run.py script allows easy execution of  Balor's graph compiler, as well as both training and inference for the GNN QoR estimator. All files needed to run each stage of the process are available [here](https://polybox.ethz.ch/index.php/s/IG0Zhi7ASMkZ12R), allowing you to immediately train a model locally, or use our pre-trained models. This is especially for those who do not wish to build our graph compiler locally. The script quickstart.py allows easy automatic download of these files to the correct locations.

#### quickstart.py

"--download_dataset" downloads >37,000 db4hls designs pre-encoded and ready for training, and places them in "balorgnn/datasets/db4hls_download/"

"--download_pretrained" downloads our model weights from epoch 580 of training, along with the validation and test results to allow our generalized scripts to select the correct model weights. These files are put in "balorgnn/outputs/model_weights/db4hls_pretrained/limerick/snake/0/" and "balorgnn/outputs/results/db4hls_pretrained/limerick/snake/0/" respectively. None of our open-sourced scripts actually read in these model weights, the evaluate functions directly reads the test set results, but these will be provided shortly.



### Methodology

Balor uses a custom graph compiler, built using [ROSE](https://github.com/rose-compiler/rose), which is tailored to both graph neural networks and the HLS process. It uses hierarchical graph neural networks to convert graph representations to graph embeddings, and then passes them to feed-forward neural networks for the actual QoR estimation.

![alt text](docs/images/flow.png)


### Balorgnn

The "balorgnn" folder contains all code relating to the GNN encoder and QoR estimators. 

It contains a "requirements.txt" for installing required python depencies. Pytorch Geometric and its dependencies are not included in "requirements.txt" as I find it's easier to directly follow their [installation instructions](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html).


Balorgnn is built to work as a local python library, and so contains a setup.py and a duplicate balorgnn folder. Like all local python libraries, it can be added to a python enviroment with:
```
pip install -e .
```

The generate/generate_dataset.py script converts c++ files to pytorch geometric .pth files, with support for 4 different datasets. (TODO: Individual instructions for each dataset)

The train/train.py script trains the GNN encoders and estimators, and performs inference on a validate and test set every 10 epochs.

### Graph Compiler

The "graph_compiler" folder contains all of the code for converting c++ code to graph representations, encoded them in the DOT graph description language from the Graphviz project. To compile it, you will need to first build [ROSE](https://github.com/rose-compiler/rose) [0.11.145.3](https://github.com/rose-compiler/rose/commit/102bc598b74b00a657510f763dabbfb18ed8bdb9) with [Boost](https://www.boost.org/) 1.67.0.

Once built, the wrapper script run_graph_compiler.py allows quick use of the compiler without specifying individual settings.

We are planning to provide a VM with the graph compiler pre-built for those who would like to experiment with it, and it will be available shortly.
