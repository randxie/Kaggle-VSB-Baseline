# Kaggle VSB Power Line Fault Detection Baseline

## Design Ideas

### Code Structure
* Keep it efficient and simple
  * Use Pytorch dataloader to do feature extraction. It should support use cases in either typical feature engineering based workflow or deep learning workflow
  * Add new feature extractor only requires inheriting the ```AbstractExtractor``` then write your ```process_fn``` and ```extract_fn```. Before computation, feature extractor will use the first signal to infer feature dimension so that you don't need to pre-specify that. Remember to update the feature name as the feature extractor will automatically do the caching for you.
* Reproducibility
  * Use json to configure parameters in the pipeline. It helps you to document your experiments. 
  * Use random seed to control data augmentation so that results are reproducible.
  
### Signal Processing
1. Remove sine pattern using a high pass filter. Based on [1], the faulty pattern exist in high frequency region so I removed frequencies under 1000 Hz.
2. Wavelet de-noising as mentioned in [1] and [2]. Afterwards, peak statistics is computed.

Note: I did not do the sine synchronization. The phase shift is used as a way to do augmentation.

## How to run

### Full training
1. Download and unzip data to "input" folder. You should have five files: metadata_test.csv, 
sample_submission.csv, train.parquet, metadata_train.csv and test.parquet.
2. Make sure you use Python 3, install pytorch by following instructions in their website and do ```pip install -r requirements.txt```.
3. To train and generate submission files, go to "src" folder and do ```python module_train_submit.py --config configs/basic.json```.
4. Find generated feature in "caching" folder and submission file in "submission" folder.

### Only generate feature
do ```python module_extract_feature --config configs/basic.json --mode=train```

### Debug mode
Set IS_DEBUG flag in ```common.py``` to True. It will use the first 100 signals to validate the whole pipeline.

## Observations
* Performance
  * Running whole pipeline in 8-core machine with multiprocessing still takes 3 hr. It is expected as I did not assume the signals have same length. Some optimization can not be done. Also, the computation is CPU based and do not leverage GPUs.
* Inaccurate cross validation score
  * With Group KFold, the cross validation score is still much higher than LB score. It could be related to feature skew in the test data or label skew. Proper detection could be done here to give us higher confident result.

## Plans
* Build better validation scheme, add skew detection.
* Continue to push forward the limit of feature engineering.
* Emulate wavelet signal processing structure to design a 1D CNN.

## Reference
[1] [A Complex Classification Approach of Partial Discharges from Covered Conductors in Real Environment](https://www.dropbox.com/s/2ltuvpw1b1ms2uu/A%20Complex%20Classification%20Approach%20of%20Partial%20Discharges%20from%20Covered%20Conductors%20in%20Real%20Environment%20%28preprint%29.pdf?dl=0)

[2] [Analysis of time series data](http://dspace.vsb.cz/bitstream/handle/10084/133114/VAN431_FEI_P1807_1801V001_2018.pdf)


