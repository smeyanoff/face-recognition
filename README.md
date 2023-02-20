# face-recognition
The program requires identify people with face image only. We need to have an ability to compare two person's photos. Moreover, we need to define people who wants to pretend to be another person and it will be fraud cases.

# Researching progress
- Choose the model 
- Get baseline 
- Try to improve it

# At this moment 
At this moment the first experiment has been done. We added the logic of random choosing persons and photos. Config parameter <FROAD_PROBA> can regulate a probability of the fraud.

# Experiment result
You can have a look on the [notebook](https://github.com/smeyanoff/face-recognition/blob/main/data/experiments/experiments_results.ipynb) with experiments.

# Fast start
At first you need to add photos to data directory. Photos need to have next structure:
```
| - data
| -- photo_folder_name
| --- person1
| ---- person1_photo1
| ---- person1_photo2
| ---- ...
| ---- person1_photoN
| --- person2
| ---- person2_photo1
| ---- person2_photo2
| ---- ...
| ---- person2_photoM
| --- ...
| --- personK
| ---- personK_photo1
| ---- personK_photo2
| ---- ...
| ---- personK_photoL
```
Then you need to change <DATA_PATH> to your photo_folder_name parametr in folder `config` / `config.py` file. 
```
python3 -m venv .venv_facerec - python3.8 has been used
source .venv_facerec/bin/activate - for linux
pip install -r requirements.txt
python main.py
 ```
 Script main.py will compute euclidean distance of two photo embandings into `data` / `experimets` folder.
