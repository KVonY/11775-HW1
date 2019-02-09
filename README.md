# 11775-HW1
## run.feature.sh
Set the video path and the cluster number (here I used cluster_num=500).
Run ./run.feature.sh to:
- extract MFCC features
- select a small portion of the MFCC vectors
- trains a k-means model
- generate k-means feature vectors
- generate the ASR-based features
## run.med.sh
Run ./run.med.sh to:
- run the following under MFCC and ASR settings
- iterate over events and train a SVM model for each of them
- apply model to test set and output the prediction score
- compute the average precision
