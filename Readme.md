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

## Experiments on synthetic data

### Running the actual experiments
To tun the experiments on synthetic data move into the scripts folder of pyRDMM.
```
cd ./pyRDMM/scripts/
```
You should now configure the `run_experiments.py` such that those experiments are executed. If you execute all the experiments be aware that running them will take about 20h.
you can run those with the virtual environment activated by
```
python run_experiments.py 
```
### Creating human readable visualizations
After all the experiments are finished the script `export_eval_results.ipynb` can be used to create plots and to export the results in a more human readable fashion.


## Experiments on real world data
We performed two experiments on real world datasets. One on a housing dataset available at kaggle and a second experiment with survey data for the European Social Survey (ESS).


### Experiments for the house price datasets
To run the house price experiments you will need to download the data manually from kaggle. The corresponding links are:
 - https://www.kaggle.com/dansbecker/melbourne-housing-snapshot and
 - https://www.kaggle.com/ruiqurm/lianjia.


To then run the experiments for the house price data consider the script `real_data_housing.ipynb` in `pyRDMM/scripts/`. There all you need to do is adjust the `folder` in the second cell to point to the locations where the downloaded files are located.


### Experiments for the European Social Survey (ESS) data 
