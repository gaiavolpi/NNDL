# Car Classification with CompCars

Neural Networks and Deep Learning Project focused on car model and make classification via ResNet architecture

## Project Structure
- `preprocessing.ipynb`: Script to clean and format the dataset. 
- `dataset.py`: Custom pytorch dataset class. 
- `utils.py` / `training_functions.py`: Support functions.
- `Car_whole.ipynb` / `Car_parts.ipynb`: Notebook to train and test the model.


## Dataset Setup

To reproduce the results, manually download the original dataset:

1. Download the **CompCars** from [MMLAB - CompCars](https://mmlab.ie.cuhk.edu.hk/datasets/comp_cars/).
2. Extract the dataset in a repository called `CompCars` in the root of this project. 

.
├── CompCars/
│   ├── data/
│   │   ├── image/
│   │   └── label/
│   └── train_test_split/
├── dataset.py
└── ...
