import pandas as pd
import json

def flatten_education_data(df):
    """
    Flatten education data in a DataFrame by iterating through each row in the DataFrame,
    extracting the education data from each row (assuming it's in a column named 'education'),
    and creating a new DataFrame with the flattened data.
    
    The education data is expected to be a list of dictionaries, where each dictionary
    represents an education entry and contains the following keys: 'degree', 'description',
    'end_year', 'field', 'institute_logo_url', 'start_year', 'title', and 'url'.
    
    :param df: The DataFrame containing the education data
    :return: A new DataFrame with the flattened education datas
    """
    # Create an empty list to store flattened data
    flattened_data = []
    
    # Iterate through each row in the DataFrame
    for index, row in df.iterrows():
        # Get the education data (assuming it's in a column named 'education')
        education_data = row['education']
        
        # If the education data is a string, convert it to a list
        if isinstance(education_data, str):
            education_data = json.loads(education_data)
        
        # If it's not a list, make it a list
        if not isinstance(education_data, list):
            education_data = [education_data]
        
        # Flatten each education entry
        for entry in education_data:
            # Create a dictionary with the flattened data
            flattened_entry = {
                'degree': entry.get('degree'),
                'description': entry.get('description'),
                'end_year': entry.get('end_year'),
                'field': entry.get('field'),
                'institute_logo_url': entry.get('institute_logo_url'),
                'start_year': entry.get('start_year'),
                'title': entry.get('title'),
                'url': entry.get('url')
            }
            
            # Add other columns from the original DataFrame
            for col in df.columns:
                if col != 'education':
                    flattened_entry[col] = row[col]
            
            flattened_data.append(flattened_entry)
    
    # Create a new DataFrame from the flattened data
    return pd.DataFrame(flattened_data) if flattened_data else pd.DataFrame()

# Example usage
if __name__ == "__main__":
    # Create a sample DataFrame
    sample_data = {
        'id': [1, 2],
        'name': ['John', 'Sarah'],
        'education': [
            [
                {'degree': "Bachelor's degree", 'end_year': '2001', 'field': 'Accounting', 'start_year': '1995', 'title': 'NYU Stern Business School'},
                {'title': 'NYU Stern School of Business', 'url': 'https://www.linkedin.com/school/nyu-stern-school-of-business/'}
            ],
            [
                {'degree': "Master's degree", 'end_year': '2010', 'field': 'Computer Science', 'start_year': '2008', 'title': 'MIT'}
            ]
        ]
    }
    
    df = pd.DataFrame(sample_data)
    
    print("\nOriginal DataFrame:")
    print(df)
    
    # Flatten the education data
    flattened_df = flatten_education_data(df)
    
    print("\nFlattened DataFrame:")
    print(flattened_df)
