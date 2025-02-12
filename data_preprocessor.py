import os
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns

from sqlalchemy import create_engine

# Function to reshape array
def prepare_dataset(data, steps, output_steps):
    X, y = [], []
    for i in range(len(data) - steps - output_steps):
        a = data[i:i + steps + output_steps]
        X.append(a)
        y.append(data[i+steps+output_steps])
    return np.array(X), np.array(y)

# Define a function to create sequences of data
def create_sequences(data, seq_length, target_col_index):
    X, y = [], []    
    for i in range(seq_length, len(data)):
        X.append(data[i-seq_length:i])
        y.append(data[i, target_col_index])  # 'the target column' is the target
    return np.array(X), np.array(y)

# Function to read data based on file extension
def read_data(file_path):
    _, file_ext = os.path.splitext(file_path)
    if file_ext == '.csv':
        return pd.read_csv(file_path)
    elif file_ext == '.json':
        return pd.read_json(file_path)
    elif file_ext in ['.xls', '.xlsx']:
        return pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported file format")
    
def read_database_data(database_url, table_name):
    engine = create_engine(database_url)    

    # Load data from the database into a DataFrame
    with engine.connect() as connection:
        query = f"SELECT * FROM {table_name}"
        df = pd.read_sql(query, con=connection)

    return df


# Check if there are duplicates in the dataset
def drop_duplicates(df, columns=None): 
	if columns == None: 
		df.drop_duplicates(inplace=True) 
	else: 
		df.drop_duplicates(subset = columns, inplace=False)
	return df 

 # Check the data for missing values
def check_missing_data(df):
    proportion_null_rows = 100*(round(df.isnull().any(axis=1).sum()/df.any(axis=1).count(),2))
    if proportion_null_rows <= 5:
        print(f"There are {df.isnull().any(axis=1).sum()} rows with a null value. All of them are erased!")
        df.dropna()
    else:
        print("Too many null values, we need to check columns by columns further.")
        if df.isnull().sum().sum() > 0:
            print("\nProportion of missing values by column")
            values = 100*(round(df.isnull().sum()/df.count(),2))
            print(values)
            dealing_missing_data(df)
        else:
            print("No missing values detected!")

# Function to deal with missing data          
def dealing_missing_data(df):
    values = 100*(round(df.isnull().sum()/df.count(),2))
    to_delete = []
    to_impute = []
    to_check = []
    for name, proportion in values.items():
        if int(proportion) == 0:
            continue
        elif int(proportion) <= 10:
            to_impute.append(name)
            df.fillna(df[name].median()) 
        else: 
            to_check.append(name)
    print(f"\nThe missing values in {to_impute} have been replaced by the median.")
    print(f"The columns {to_check} should be further understood")
    
# Function to check data types
def check_data_types(df, expected_types):
    """
    Check the data types of a DataFrame against expected types.

    Parameters:
    - df (pd.DataFrame): The DataFrame to check.
    - expected_types (dict): A dictionary mapping column names to expected data types (e.g., 'int', 'float', 'datetime').

    Returns:
    - dict: A report of mismatches and suggested corrections.
    """
    for column, expected_type in expected_types.items():
        actual_type = df[column].dtype

        # Create a readable version of numpy dtype for reporting
        readable_type = np.dtype(actual_type).name
        if not np.issubdtype(actual_type, np.dtype(expected_type).type):
            message = f"Column '{column}' has type '{readable_type}' instead of '{expected_type}'."
            suggestion = f"Convert '{column}' to '{expected_type}'."
            print(f"{message}", f"{suggestion}")

    print("No data types mismatch detected")

# Function to find outliers using IQR
def find_outliers_IQR(df):
    outlier_indices = []
    df = df.select_dtypes(include=['number'])
    for column in df.columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Get the indices of outliers for feature column
        outlier_list_col = df[(df[column] < lower_bound) | (df[column] > upper_bound)].index
        outlier_indices.extend(outlier_list_col)
    
    outlier_indices = list(set(outlier_indices))  # Get unique indices
    return df.iloc[outlier_indices]

# Visualize the data with outliers using scatter plot and box plot
def visualise_outliers(data, column_name, outliers=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Scatter plot
    #ax1.scatter(range(len(data)), data[column_name], c=['blue' if not x else 'red' for x in outliers[column_name]])
    ax1.scatter(range(len(outliers)), outliers[column_name], c=['blue' if not x else 'red' for x in outliers[column_name]])
   
    ax1.set_title('Dataset with Outliers Highlighted (Scatter Plot)')
    ax1.set_xlabel('Index')
    ax1.set_ylabel('Value')

    # Box plot
    sns.boxplot(x=outliers[column_name], ax=ax2)
    ax2.set_title('Dataset with Outliers (Box Plot)')
    ax2.set_xlabel('Value')

    plt.tight_layout()
    plt.show()

# Function to detect outliers using IQR
def detect_outliers_iqr(data):
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return (data < lower_bound) | (data > upper_bound)

# Remove outliers
def remove_outliers(data, column_name, outliers):
    data_cleaned = data[column_name][~outliers]

    print(f"Original dataset size: {len(data)}")
    print(f"Cleaned dataset size: {len(data_cleaned)}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Scatter plot
    ax1.scatter(range(len(data_cleaned)), data_cleaned[column_name])
    ax1.set_title('Dataset After Removing Outliers (Scatter Plot)')
    ax1.set_xlabel('Index')
    ax1.set_ylabel('Value')

    # Box plot
    sns.boxplot(x=data_cleaned[column_name], ax=ax2)
    ax2.set_title('Dataset After Removing Outliers (Box Plot)')
    ax2.set_xlabel(column_name)

    plt.tight_layout()
    plt.show()

# Function to cap outliers
def cap_outliers(data, lower_percentile=5, upper_percentile=95):
    lower_limit = np.percentile(data, lower_percentile)
    upper_limit = np.percentile(data, upper_percentile)
    return np.clip(data, lower_limit, upper_limit)

# Function to visualize the correlation matrix using a heatmap
def visualize_correlation_matrix(df):
    correlation = df.corr()
    sns.heatmap(
        correlation,
        vmin=-1, vmax=1, center=0,
        cmap=sns.diverging_palette(50, 500, n=500),
        square=True
    )
    plt.show()  

def invert_transform(data, shape, column_index, scaler):
    dummy_array = np.zeros((len(data), shape))    
    dummy_array[:, column_index] = data    
    return scaler.inverse_transform(dummy_array)[:, column_index]