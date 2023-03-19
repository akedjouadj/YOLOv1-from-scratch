### Yolov1 model from scratch

##### I'm convinced that the best way to understand a machine learning model is to code it from scratch and that was the principal interest of this project.

- You can read the original paper via the link : https://arxiv.org/abs/1506.02640

- This repo contains my implementation of the first version of the model YOLO (You only look once) for object detection. I was inspired by Aladdin Persson's video available on youtube via the link : https://www.youtube.com/watch?v=n9_XyCGr-MI&list=PLhhyoLH6Ijfw0TpCTVTNk42NN08H6UvNq&index=5

- Make sure you install the requirements before starting. You can create a venv and then execute this command in your terminal : pip install -r requirements.txt

- I train my model on the PASCAL VOC dataset (see the paper) available on Kaggle via the link : https://www.kaggle.com/datasets/734b7bcb7ef13a045cbdd007a3c19874c2586ed0b02b4afc86126e89d00af8d2?resource=download. Because I don't have GPU and a lot of vRAM, I just trained my model on the 100examples.csv file available in the dataset on google colab (2CPU, 12Gb vRAM). The model learns well and quickly reaches a mAP greater than 0.9. The notebook that trained it on google colab is available in the repo. If you have more computing power and want to train the model on more data, you can take the train.py code and make some adjustments.

- The model_application_prediction_and_visualization.ipynb notebook can help you to test your trained model and make a quickly visualization of the results. I used the pre-trained model saved on my computer, unfortunately I couldn't load it on the repo for memory reasons (1Gb size). Obviously, YOLO is a very big model (see the paper)
