# CS_7457-001-Final-project Group 1

## Performace test

The performance test scripts are located in the performance test folder, the plot tool is the drawing tool, and the raw data from the test is also in there.

### Script Explanation

vpn_band.py - For VPN bandwidth testing

vpn_lj.py - Tests for VPN latency and jitter

## Privacy test

The privacy test contains only the training part of the model and the test of the model. New and old data are stored in dataset as a zip package. The trained model is located in models.
The raw traffic data can be downloaded at the link provided in the txt.

### Script Explanation

CNN_LSTM.py - Model Training

Model_Test.py - Model Testing, result are confusion matrixs

## Other tools in paper

  - [Traffic(.pcap) to Png](https://github.com/yungshenglu/USTC-TK2016.git) - Used convert network traffic to images
  - [NDPI](https://github.com/ntop/nDPI.git) - Tools for checking network traffic
  - [Old dataset](https://www.unb.ca/cic/datasets/vpn.html) 
