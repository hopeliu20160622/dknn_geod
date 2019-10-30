# reachability packages
conda create --name dknn_geod python=3.6
source ~/anaconda3/etc/profile.d/conda.sh
conda activate dknn_geod
conda install -c conda-forge matplotlib
conda install -c anaconda scikit-learn

pip install --upgrade pip
pip install tensorflow==1.15.0
#pip install tensor2tensor==1.14.1
#t2t-datagen --generate_data --data_dir=~/t2t_data --problem=image_cifar10_plain
#t2t-datagen --generate_data --data_dir=~/t2t_data --problem=image_cifar100_plain

conda install -c conda-forge jupyterlab
ipython kernel install --user --name=dknn_geod
conda install -c anaconda pandas
pip install cleverhans
conda install -c akode falconn
conda install pylint
conda deactivate
