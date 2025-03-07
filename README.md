# No Code Data Science Framework

This repository contains a Streamlit application that provides a no-code data science framework. It allows users to upload CSV or Excel files, explore the data, and automatically train machine learning models based on the selected target variable.

## Features

-   **Data Upload and Exploration:**
    -      Upload CSV and Excel files.
    -      Display the first few rows of the data.
    -      View summary statistics and information.
    -      Check the dimensions of the file.
    -      Identify empty records.
-   **Automated Model Selection:**
    -      Automatically detects the type of target variable (numerical or categorical).
    -      For numerical targets:
        -      Distinguishes between continuous and discrete targets.
        -      Trains and evaluates linear regression, polynomial regression, ridge regression, lasso regression, elastic net regression, decision tree regression, random forest regression, gradient boosting regression, and KNN regression.
    -      For categorical targets:
        -      Distinguishes between binary and multi-class classification.
        -      Trains and evaluates logistic regression, decision tree classification, random forest classification, gradient boosting classification, and KNN classification.
-   **Simple User Interface:**
    -      Uses Streamlit for an interactive and easy-to-use web interface.
    -   Allows for easy file uploading and target variable selection.
-   **Basic Data Cleaning:**
    -   Exclusion of ID and Name columns.
-   **Model Evaluation:**
    -   Mean squared error for regression models.
    -   Accuracy score for classification models.

## Getting Started

### Prerequisites

-      Python 3.6+
-      Streamlit
-      Pandas
-      Scikit-learn
-   Numpy

### Installation

1.  Clone the repository:

    ```bash
    git clone [https://github.com/your-username/no-code-data-science-framework.git](https://www.google.com/search?q=https://github.com/your-username/no-code-data-science-framework.git)
    cd no-code-data-science-framework
    ```

2.  Install the required packages:

    ```bash
    pip install streamlit pandas scikit-learn numpy openpyxl
    ```

### Usage

1.  Run the Streamlit application:

    ```bash
    streamlit run app.py
    ```

2.  Open the application in your web browser (usually `http://localhost:8501`).

3.  Upload your CSV or Excel file.

4.  Explore the data using the provided options.

5.  Select the target variable.

6.  The application will automatically train and evaluate the appropriate machine learning models.

## File Structure
no-code-data-science-framework/
├── app.py          # Main Streamlit application
├── README.md       # This file


## Code Explanation

-   `app.py`: Contains the Streamlit application code, including data loading, exploration, model training, and evaluation.
-   The script uses scikit-learn for machine learning models and pandas for data manipulation.
-   Streamlit is used to create the interactive web interface.
-   The code includes functions for various regression and classification models, as well as helper functions for data analysis and target variable detection.

## Future Improvements

-   Add more data preprocessing options (e.g., handling missing values, feature scaling).
-   Implement hyperparameter tuning for the models.
-   Provide more detailed model evaluation metrics.
-   Add visualization of the model results.
-   Allow users to download the trained models.
-   Add more error handling and user feedback.
-   Add model persistance (pickle).
-   Add support for more file types.
-   Allow user to select test size.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue.

## License

This project is licensed under the [MIT License](LICENSE).
