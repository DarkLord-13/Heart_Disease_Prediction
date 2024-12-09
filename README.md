# Heart Disease Prediction

This project demonstrates a machine learning approach to classify whether a patient has heart disease or not using a logistic regression model.

## Project Overview

The project involves the following steps:
1. **Data Loading**:
    - Load the heart disease dataset.
2. **Data Preprocessing**:
    - Check for missing values and handle them if necessary.
    - Split the dataset into features and labels.
3. **Model Training**:
    - Train a logistic regression model on the training data.
4. **Model Evaluation**:
    - Evaluate the model using accuracy score on both training and test data.
5. **Prediction System**:
    - Create a system to predict heart disease based on input features.

## Dependencies

The project requires the following dependencies:
- Python 3.x
- NumPy
- Pandas
- Seaborn
- Scikit-learn

## Installation

1. Clone this repository:
    ```sh
    git clone https://github.com/DarkLord-13/Machine-Learning-01.git
    ```

2. Navigate to the project directory:
    ```sh
    cd Machine-Learning-01
    ```

3. Install the required packages:
    ```sh
    pip install numpy pandas seaborn scikit-learn
    ```

## Usage

1. **Load the Data**:
    - Load the heart disease dataset.
    ```python
    import pandas as pd
    dataset = pd.read_csv('/content/heart_disease_data.csv')
    ```

2. **Data Preprocessing**:
    - Check for missing values and split the data into features and labels.
    ```python
    x = dataset.drop(columns='target', axis=1)
    y = dataset['target']
    ```

3. **Train the Model**:
    - Split the data into training and test sets, and train the logistic regression model.
    ```python
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=2, stratify=y)
    model = LogisticRegression()
    model.fit(x_train, y_train)
    ```

4. **Evaluate the Model**:
    - Evaluate the model's performance using accuracy score.
    ```python
    from sklearn.metrics import accuracy_score

    x_train_prediction = model.predict(x_train)
    training_data_accuracy = accuracy_score(x_train_prediction, y_train)

    x_test_prediction = model.predict(x_test)
    test_data_accuracy = accuracy_score(x_test_prediction, y_test)
    ```

5. **Prediction System**:
    - Create a system to predict heart disease based on input features.
    ```python
    import numpy as np

    input_data = (53, 0, 0, 130, 264, 0, 0, 143, 0, 0.4, 1, 0, 2)
    input_data = np.asarray(input_data)
    input_data = input_data.reshape(1, -1)
    prediction = model.predict(input_data)
    print(prediction)
    ```

## License

This project is licensed under the MIT License.
