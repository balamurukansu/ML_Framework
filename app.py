import streamlit as st
import pandas as pd
import numpy as np
import io
import pickle

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score

# Set the page configuration
st.set_page_config(page_title='No Code Data Science Framework', layout="wide")
st.markdown(f'<h1 style="text-align: center;">No Code Data Science / Framework</h1>', unsafe_allow_html=True)

#Check Target Column Type
def check_column_type(df, target):
    if np.issubdtype(df[target].dtype, np.number):
        return 'numerical'
    else:
        return 'categorical'

# Check if Numerical Target is Continuous or Discrete
def check_numerical_type(df, target):
    unique_values = df[target].nunique()
    if unique_values > 20:
        return 'continuous'
    else:
        return 'discrete'

#Perform Linear Regression
def linear_regression(X_train, y_train, X_test, y_test):
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    error = mean_squared_error(y_test, predictions)
    return error, predictions

#Perform Polynomial Regression
def polynomial_regression(X_train, y_train, X_test, y_test, degree=2):
    from sklearn.preprocessing import PolynomialFeatures
    poly = PolynomialFeatures(degree=degree)
    X_poly_train = poly.fit_transform(X_train)
    X_poly_test = poly.transform(X_test)
    model = LinearRegression()
    model.fit(X_poly_train, y_train)
    predictions = model.predict(X_poly_test)
    error = mean_squared_error(y_test, predictions)
    return error, predictions

#Perform Ridge Regression
def ridge_regression(X_train, y_train, X_test, y_test):
    model = Ridge()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    error = mean_squared_error(y_test, predictions)
    return error, predictions

#Perform Lasso Regression
def lasso_regression(X_train, y_train, X_test, y_test):
    model = Lasso()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    error = mean_squared_error(y_test, predictions)
    return error, predictions

#Perform Elastic Net
def elastic_net_regression(X_train, y_train, X_test, y_test):
    model = ElasticNet()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    error = mean_squared_error(y_test, predictions)
    return error, predictions

#Choose Non-Linear Models (For Discrete Numerical Target)
def decision_tree_regressor(X_train, y_train, X_test, y_test):
    model = DecisionTreeRegressor()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    error = mean_squared_error(y_test, predictions)
    return error, predictions

#Perform Random Forest Regression
def random_forest_regressor(X_train, y_train, X_test, y_test):
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    error = mean_squared_error(y_test, predictions)
    return error, predictions

#Perform Extreme Gradient Boosting Regression
def gradient_boosting_regressor(X_train, y_train, X_test, y_test):
    model = GradientBoostingRegressor()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    error = mean_squared_error(y_test, predictions)
    return error, predictions

#Perform KNN Regression
def knn_regressor(X_train, y_train, X_test, y_test):
    model = KNeighborsRegressor()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    error = mean_squared_error(y_test, predictions)
    return error, predictions

#Check if Categorical Target is Binary or Multi-Class
def check_categorical_type(df, target):
    unique_values = df[target].nunique()
    if unique_values == 2:
        return 'binary'
    else:
        return 'multi-class'

#Perform Logistic Regression
def logistic_regression(X_train, y_train, X_test, y_test):
    model = LogisticRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    return accuracy, predictions

#Perform Decision Tree Classification
def decision_tree_classifier(X_train, y_train, X_test, y_test):
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    return accuracy, predictions

#Perform Random Forest Classification
def random_forest_classifier(X_train, y_train, X_test, y_test):
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    return accuracy, predictions

#Perform Extreme Gradient Boosting Classification
def gradient_boosting_classifier(X_train, y_train, X_test, y_test):
    model = GradientBoostingClassifier()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    return accuracy, predictions

#Perform KNN Classification
def knn_classifier(X_train, y_train, X_test, y_test):
    model = KNeighborsClassifier()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    return accuracy, predictions

def check_numeric_type(column):
    try:
        # Convert the column to numeric, setting errors='coerce' will replace non-numeric values with NaN
        numeric_column = pd.to_numeric(column, errors='coerce')
        # Check if any NaN values exist in the column
        if numeric_column.isna().any():
            return (False, 'non-numeric')
        else:
            # Check if all values are integers
            if (numeric_column == numeric_column.astype(int)).all():
                return (True, 'discrete')
            else:
                return (True, 'continuous')
    except:
        return (False, 'non-numeric') 

if __name__ == "__main__":
    st.write('')
    st.write("""# Step 1: File Picker""")
    
    # Create a file uploader widget
    uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx"])
    
    df = None

    # Check if a file is uploaded
    if uploaded_file is not None:
        # Load the uploaded file into a DataFrame
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file)
    
    # Display the DataFrame if it is not None
    if df is not None:
        options_sb = ['Display the first few rows','Summary of structure and information','View basic statistical details','View dimension of file','Identify empty records']
        option = st.selectbox(label='',options=options_sb)
        if option == 'Display the first few rows':
            st.table(df.head())
        elif option == 'Summary of structure and information':
            st.write(df.info())
            buffer = io.StringIO()
            df.info(buf=buffer)
            info = buffer.getvalue()
            st.text(info)
        elif option == 'View basic statistical details':
            st.table(df.describe())
        elif option == 'View dimension of file':
            st.table(df.shape)
        elif option == 'Identify empty records':
            st.table(df.isnull().sum())

        #Create a list of exculuded columns
        exclude_columns = ['ID','Name']

        #Filter dataframe's columns to exclude specified columns
        filtered_columns = [col for col in df.columns if not any(substring in col for substring in exclude_columns)]
        
        st.write("""# Step 2: Choose Target""")
        target = st.selectbox(label='',options=filtered_columns)
        target_type = check_column_type(df, target)

        X = df.drop(columns=[target])
        y = df[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        if target_type == 'numerical':
            num_type = check_numerical_type(df, target)
            if num_type == 'continuous':
                error, _ = linear_regression(X_train, y_train, X_test, y_test)
                if error > 15:
                    error, _ = polynomial_regression(X_train, y_train, X_test, y_test)
                    if error > 15:
                        error, _ = ridge_regression(X_train, y_train, X_test, y_test)
                        if error > 15:
                            error, _ = lasso_regression(X_train, y_train, X_test, y_test)
                            if error > 15:
                                error, _ = elastic_net_regression(X_train, y_train, X_test, y_test)
            else:
                error, _ = decision_tree_regressor(X_train, y_train, X_test, y_test)
                if error > 15:
                    error, _ = random_forest_regressor(X_train, y_train, X_test, y_test)
                    if error > 15:
                        error, _ = gradient_boosting_regressor(X_train, y_train, X_test, y_test)
                        if error > 15:
                            error, _ = knn_regressor(X_train, y_train, X_test, y_test)
        else:
            cat_type = check_categorical_type(df, target)
            if cat_type == 'binary':
                accuracy, _ = logistic_regression(X_train, y_train, X_test, y_test)
            else:
                accuracy, _ = decision_tree_classifier(X_train, y_train, X_test, y_test)
                if accuracy < 0.85:
                    accuracy, _ = random_forest_classifier(X_train, y_train, X_test, y_test)
                    if accuracy < 0.85:
                        accuracy, _ = gradient_boosting_classifier(X_train, y_train, X_test, y_test)
                        if accuracy < 0.85:
                            accuracy, _ = knn_classifier(X_train, y_train, X_test, y_test)