# Redescription model mining

This package brings redescription model mining to python.

## Installation
Get some python packages
```
git clone https://github.com/Feelx234/pyRDMM.git
git clone https://github.com/flemmerich/pysubgroup.git
```
Create virtual environment
```
python -m venv env
source ./env/bin/activate
pip install -e pysubgroup/
pip install -e pyRDMM/
pip instal jupyter
```

## Running the experiments on synthetic data

To tun the experiments on synthetic data move into the scripts folder of pyRDMM.
```
cd ./pyRDMM/scripts/
```
You should now configure the `run_experiments.py` such that those experiments are executed. If you execute all the experiments be aware that running them will take about 20h.
you can run those with the virtual environment activated by
```
python run_experiments.py 
```

## Running the experiments on real world data
For the experiments with real world data you first need the data.
### Experiments for the house price datasets
The house price data we used are available at https://www.kaggle.com/dansbecker/melbourne-housing-snapshot and https://www.kaggle.com/ruiqurm/lianjia.
To then run the experiments for the house price data consider the script `real_data_housing.ipynb` in `pyRDMM/scripts/`. There all you need to do is adjust the `folder` in the second cell to point to the locations where the downloaded files are located.

