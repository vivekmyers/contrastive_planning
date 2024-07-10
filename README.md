# Contrastive Planning

Code for the paper [Inference via Interpolation: Contrastive Representations Provably Enable Planning and Inference](https://arxiv.org/abs/2403.04082).

## Installation

1. Check conda is installed and mujoco200 binaries are in path:
```
test -z "$CONDA_PREFIX" && wget -O Miniforge3.sh "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh" && bash Miniforge3.sh -b -p "$HOME/conda" && rm Miniforge3.sh 
"$HOME/conda/bin/conda" init "$(basename $SHELL)"
test ! -e ~/.mujoco/mujoco200 && mkdir -p ~/.mujoco && wget -O mujoco200.zip https://www.roboti.us/download/mujoco200_$(test $(uname) == "Linux" && echo linux || echo macos).zip && unzip mujoco200.zip && rm mujoco200.zip && mv mujoco200* ~/.mujoco
echo "$LD_LIBRARY_PATH" | grep -q mujoco200 || echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco200/bin' >> ~/.bashrc
```

2. Clone repo and build environment:
```
git clone https://github.com/vivekmyers/contrastive_planning.git
cd contrastive_planning
conda env create
conda activate contrastive_planning
```


## Running experiments

Run the following commands to train the method and baselines discussed in the paper for a single initialization seed and dataset shuffle:
```
python run_train.py --env maze2d-large-v1
python run_train.py --env door-human-v0
```
To evaluate success rates by distance and planning MSE for our method and baselines, run the following.
```
python run_eval.py --env maze2d-large-v1 --n_wypt 20 --rollout
python run_eval.py --env door-human-v0 --mse --n_wypt 5
python run_eval.py --env door-human-v0 --mse --n_wypt 1
```

The following commands will plot the results as shown in the paper.
```
python run_plot.py --env maze2d-large-v1 --rollout 
python run_plot.py --env door-human-v0 --waypoint_mse --n_wypt 5
python run_plot.py --env door-human-v0 --barplot --n_wypt 1 
python run_plot.py --env door-human-v0 --plot_plan 
```

## Reproducing Results
All quantitative results from the paper can be reproduced by running `make all`. This will train and evaluate 100 seeds by default.
