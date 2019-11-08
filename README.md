# Video Understanding
This repository is being maintained by project Team 6 from DDL course (CS 8803), Georgia Tech. We are working on Video Understading wherein given a video, we would like to predict multiple labels/categories/tags for each video. With an increase in generation and consumption of videos, efficient retrieval of videos has become a challenging task. By providing multiple labels/categories to each video, we hope to develop a better video understanding, as a step towards solving this problem. 

Our directory structure is as follows:
- code:
  - NetDissect: Contains files for Network Dissection along with README.
  - youtube8m: Contains files for YouTube 8M Video Understanding along with README.
  - Starter_Explore_sample_data.ipynb: Code for data exploration and pre-processing. The sample data and files needed for running should be downloaded in ../sample_input and can be accessed here : https://drive.google.com/drive/folders/1e_moD2WWSDyKC4LBpP_ta9ctUoVAN4La?usp=sharing  
  - LSTM.ipynb: This is a Google Colab notebook for training LSTM model for Frame-level Video data. To run this notebook, upload the youtube8m folder to Google drive, open the notebook in Google Colaboratory and run all cells. 
  - LogisticModel.ipynb: This is a Google Colab notebook for training Logistic model for Frame-level Video data. To run this notebook, upload the youtube8m folder to Google drive, open the notebook in Google Colaboratory and run all cells. 
  - Training_Resnet.ipynb: Designed and Implemented our own Residual block architectures from scratch for Video level classification. The model architecture code can be found in 'video_level_models.py' in youtube8m folder. Training_Resnet.ipynb is a Google Colab notebook for training this Resnet model for Video-level data. To run this notebook, upload the youtube8m folder to Google drive, open the notebook in Google Colaboratory and run all cells.
- docs: Contains project proposal, progress reports and presentation.

Installing Dependencies:
To install all the dependencies for this project, run the following command:
```
pip install -r requirements.txt
```


Team Members:
- Sanmathi Kamath
- Pranjali Kokare
- Alekhya Munagala
