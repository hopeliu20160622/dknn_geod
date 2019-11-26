# reachability packages
conda create --name gdknn python=3.6
source ~/anaconda3/etc/profile.d/conda.sh
#source ~/miniconda/etc/profile.d/conda.sh
conda activate gdknn

conda install -c anaconda pyyaml
conda install -c anaconda numpy==1.16.1
conda install -c anaconda scipy==1.2.1
conda install -c anaconda scikit-learn==0.21.3
conda install -c anaconda pandas==0.25.2
conda install -c conda-forge matplotlib==3.1.1
conda install -c anaconda seaborn==0.9.0

conda install -c anaconda tensorflow=1.13.1
#conda install -c anaconda tensorflow-gpu=1.13.1

conda install -c conda-forge jupyterlab
ipython kernel install --user --name=gdknn
conda install -c anaconda pylint
conda install -c anaconda pyyaml
pip install cleverhans
conda install -c akode falconn
conda install -c anaconda cython
conda install -c conda-forge keras==2.2.4
cd src/
python setup_fast_gknn.py build_ext --inplace
cd ../
conda deactivate

mkdir results/MNIST/
mkdir results/CIFAR10/
