# Realistic-looking-rainfall-forecasts
MeteoSwiss project for ETH Zürich Data Science Lab 2020

### Usage
Both CGAN and GAN models can be trained using `src/main.py` like this:
```cd src; python3 main.py --cuda --devices 0```


### Contributing
Before pushing code to master, please ensure that tests pass using 
```cd src; export PYTHONPATH=. ; pytest ./tests.py```
