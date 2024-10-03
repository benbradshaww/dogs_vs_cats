# Dogs vs Cats Classifier using EfficientNet

This repository contains a deep learning model for classifying images of dogs and cats using the EfficientNet-B0 architecture. The model is trained on the popular Dogs vs Cats dataset and uses transfer learning with the EfficientNet architecture to achieve high accuracy.

## Table of Contents

	1.	Project Overview
	2.	Dataset
	3.	Model Architecture
	4.	Installation
	5.	Training
	6.	Evaluation
	7.	Results
	8.	Usage
	9.	Contributing
	10.	License

## Project Overview

This project will implement my own version of the EfficientNet-B0 model to classify dog and cat images.
For more details on the model I implemented. Please visit: https://arxiv.org/pdf/1905.11946.

The goal of this project is to improve my skills with:
* Creating Pytorch vision models.
* ML training pipelines.
* Implementations of pre-commits (some custom made).
* Employing github actions.
* Learn about training on virutal GPUs.

## Dataset

The dataset used in this project is the Dogs vs Cats dataset, which contains 8000 labeled images of dogs and cats for training.
You can download the dataset from Kaggle. The instructions will be provided below.

## Data Structure:

## Model Architecture


Key Features:

	•	EfficientNet-B0 as the base model.
	•	A custom classification head with a fully connected layer and softmax activation.
	•	Data augmentations (e.g., random rotations, horizontal flips) applied to the training set.

## Installation

To set up this project locally, follow these steps:

### 1. Clone the Repository:
```
git clone https://github.com/your-username/dogs-vs-cats-efficientnet.git
cd dogs-vs-cats-efficientnet
```
### 2. Set Up the Virtual Environment:
```
python3 -m venv venv
# For Mac:
source venv/bin/activate
# For Windows:
venv\Scripts\activate
```
### 3. Install Dependencies:
pip install -r requirements.txt

### 4. Download the Dataset:

- Download the dataset from [Kaggle](https://www.kaggle.com/c/dogs-vs-cats/data).
- Unzip it and place the files into a `data/` directory as shown in the **Data Structure** section.

Alternatively, if you have a valid Kaggle API in your home folder, run the function `download_data` to get the necessary data.

## Training

Run the
