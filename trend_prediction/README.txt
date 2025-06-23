To preprocess trend segments for sigle time series RUN: python preprocessing/preproprecessor.py JSE (Specific column can be update in the script)

TO preprocess trend segments for multi-variate time series RUN: python preprocessing/prepare_data.py --dataset jse 

Prediction:
Using lstm model with trend features: python execute.py --dataset jse --feature trend --algorithm lstm --nruns 1 --verbose --save

Using lstm model with point data features: python execute.py --dataset jse --feature pointdata --algorithm lstm --nruns 1 --verbose --save 


STGNN:
GWN with trend features: python -m stgnn.run.stgnn_main --dataset jse --feature trend --algorithm gwn --nruns 1 --verbose
