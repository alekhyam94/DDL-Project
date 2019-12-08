# Video Understanding
This repository is being maintained by project Team 6 from DDL course (CS 8803), Georgia Tech. We are working on Video Understading wherein given a video, we would like to predict multiple labels/categories/tags for each video. With an increase in generation and consumption of videos, efficient retrieval of videos has become a challenging task. By providing multiple labels/categories to each video, we hope to develop a better video understanding, as a step towards solving this problem. 

### Example of Tagging a Video
![Example of tagging a video. Suitable tags: Celebration, Birthday, Blowing Candles, Cake](video_tagging.gif)
<br />  Tags: Celebration, Birthday, Blowing Candles, Cake

### Installation
To install all the dependencies for this project, run the following command:
```
pip install -r requirements.txt
```
To know more about third-party code, please refer to README of individual folders of each of those sections.

### Code Structure
- code:
  - youtube8m: Contains files for YouTube 8M Video Understanding along with README.
  - Starter_Explore_sample_data.ipynb: Code for data exploration and pre-processing. The sample data and files needed for running should be downloaded in ../sample_input and can be accessed here : https://drive.google.com/drive/folders/1e_moD2WWSDyKC4LBpP_ta9ctUoVAN4La?usp=sharing  
  - LSTM.ipynb: This is a Google Colab notebook for training LSTM model for Frame-level Video data.  
  - LogisticModel.ipynb: This is a Google Colab notebook for training Logistic model for Frame-level Video data. 
  - Training_Resnet.ipynb: Designed and Implemented our own Residual block architectures from scratch for Video level classification. The model architecture code can be found in 'video_level_models.py' in youtube8m folder. Training_Resnet.ipynb is a Google Colab notebook for training this Resnet model for Video-level data.
  - ResNetLayers.ipynb: This architecture contains 2 modules, mini-residual block and basic-residual block for video-level classification.  The model architecture code can be found in 'video_level_models.py' in youtube8m folder.

- docs: Contains project proposal, progress reports and presentations.

### Notes
- To run these notebooks, upload the youtube8m folder to Google drive, open the notebook in Google Colaboratory and run all cells. 
- All the models have been built using TensorFlow. In TensorFlow, training is done in two phases. In the first phase, the computation graphs are built and in the second phase, actual training is done using the graph. To know more, refer: https://deepnotes.io/tensorflow
- The accuracies in the notebook may not reflect our best model performance. Please refer to reports in /docs for performance evaluation.  

### Eva Testing
We have also contributed to Eva by improving coverage of Query Parser module. Please refer to our work at this repository:
https://github.com/KokareIITP/Eva

### Contributors
- Sanmathi Kamath
- Pranjali Kokare
- Alekhya Munagala
