from sklearn.model_selection import train_test_split
import pandas as pd
import os

# Replace data


def replace_special_chars(x):
    return x.replace("&", " & ").replace("=", " = ").replace("?", " ? ").replace(":", " : ")


def merge_csv_files(input_folder, output_file):
    dataframes = []

    # Get a list of all CSV files in the input directory
    for folder in os.listdir(input_folder):
        folder_path = os.path.join(input_folder, folder)
        if (os.path.isdir(folder_path)):  # Check if the folder is a directory
            for file_path in os.listdir(folder_path):
                if (file_path.endswith('.csv')):
                    print(file_path)
                    df = pd.read_csv(os.path.join(
                        folder_path, file_path), sep="|")
                    print(df.head(10))
                    dataframes.append(df)

    # Merge all DataFrames into a single DataFrame
    merged_df = pd.concat(dataframes, ignore_index=True)

    # Save the merged DataFrame to a new CSV file
    merged_df.to_csv(output_file, index=False, sep="|")


def read_and_merge_csv(input_file, output_file):
    df = pd.read_csv(input_file, sep="|")

    # Replace NaN values in the response column with ''
    df['response'] = df['response'].fillna('')

    # Assume the request and response columns are named 'request' and 'response'
    df['value'] = df['request'] + ' ' + df['response']

    # Keep only the necessary columns
    df = df[['date_time', 'time_exec', 'label', 'value']]

    print(df.head(10))

    # Save the processed DataFrame to a new CSV file
    df.to_csv(output_file, index=False, sep="|")


def split_train_test(input_file, train_file, test_file):
    df = pd.read_csv(input_file, sep="|")

    df.value = df.value.map(lambda x: replace_special_chars(
        x) if isinstance(x, str) else x)

    print(df.head(10))

    # Split the dataset into training and testing sets
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=43)

    # Save the training and testing sets to CSV files
    train_df.to_csv(train_file, index=False, sep="|")
    test_df.to_csv(test_file, index=False, sep="|")


if __name__ == "__main__":
    output_folder = r"C:\Users\MSI\Desktop\RA\MyTvAnamalyDetection\log_mytv_anomaly\data_training\log_api"
    # output_folder = r'/u01/data/mytv_fd/cuongdd/data_train/log_api/'

    # # Use merge csv files
    # merge_csv_files(output_folder, output_folder + 'log_api_all.csv')

    # read_and_merge_csv(output_folder + 'log_api_all.csv',
    #                    output_folder + 'log_api_all_final.csv')

    # Call the function to split train and test datasets
    split_train_test(output_folder + r'\log_api_all_final.csv',
                     output_folder + r'\train_data.csv',
                     output_folder + r'\test_data.csv')
