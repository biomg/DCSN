# Double convolutional stitching networks for prostate cancer patients state prediction
In this work, we developed a deep learning model called DCSN, which can predict the prostate status of patients by processing the number of gene mutations in patients. We use CNN to improve the prediction accuracy of prostate cancer status. Unlike traditional CNN, DCSN takes the number of input eigenvalues as the number of channels, and uses two CNN1Ds with different channel numbers to process the input, then splice the results into the full connection layer to get the final prediction results, so as to extract features of different dimensions and improve accuracy. Through experiments, we prove that DCSN can get a more competitive result. During the experiment, we found that some minor changes in parameters will have a greater impact on the results of DCSN. In the next work, we will try to make more subtle adjustments to the parameters of DCSN, and introduce some cutting-edge over fitting mitigation methods to get better results, and apply DCSN to other cancer state prediction experiments.
# Dependency:
Python 3.7
Pytorch 1.12.1
numpy 1.18.5
scikit-learn 0.23.2
# Supported GPUs
Now it supports GPUs. The code support GPUs and CPUs, it automatically check whether you server install GPU or not, it will proritize using the GPUs if there exist GPUs.
In addition, WVDL can also be adapted to protein binding sites on DNAs and identify DNA binding speciticity of proteins.
It supports model training, testing.
# Usage:
python DCSN.py
# Contact
Junjiang Liu:junjiang.liu@foxmail.com
