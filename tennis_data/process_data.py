import pandas as pd
import os

def process_data(data_path, output_path):
    """Reads all excel files in a directory, combines them, and saves them as a CSV."""
    all_files = [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith(('.xls', '.xlsx'))]
    
    all_data = []
    for f in all_files:
        try:
            if f.endswith('.xlsx'):
                df = pd.read_excel(f, engine='openpyxl')
            else:
                df = pd.read_excel(f)
            all_data.append(df)
        except Exception as e:
            print(f"Error reading {f}: {e}")
            if '2012' in f:
                print(f"Skipping file {f} due to read error.")
                continue

    combined_df = pd.concat(all_data, ignore_index=True)
    combined_df.to_csv(output_path, index=False)

if __name__ == '__main__':
    data_path = 'C:/Users/Carlos/Documents/ODST/tennis_data'
    output_path = 'C:/Users/Carlos/Documents/ODST/tennis_data/tennis_data.csv'
    process_data(data_path, output_path)
    print(f"Data processed and saved to {output_path}")