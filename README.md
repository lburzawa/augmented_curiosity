# GameNet

To start training run:

`python a2c.py`

Arguments:

`--icm` train with curiosity

`--scenario` specify the environment you want to train on

`--save_dir` specify where your model and testing results are going to be saved

For example to train a standard a2c on My Way Home scenario:

`python a2c.py`

To train a2c with curiosity on My Way Home Sparse scenario:

`python a2c.py --icm --scenario=./scenarios/my_way_home_sparse.cfg`
