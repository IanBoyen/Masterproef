import pandas as pd

# Define Excel file path
excel_file = "../../Types_Cutting_Inserts.xlsx"
csv_file = "Wear_data.csv"
insert_type_list = ["RM121263NE-BB", "RM090955NE-AB", "RM090955NE-AC", "RM121279NE-CV", "RM121279NE-DF", "RM121279NE-CU", "SNC-44-170", "SNC-44-60KH04"]
# Read all sheets into a dictionary of DataFrames
sheets = pd.read_excel(excel_file, sheet_name=None)

# Create an empty list to store data
data_list = []

# Process each sheet
for insert_type, df in sheets.items():
    if insert_type in insert_type_list:
        df = df.iloc[:, :5] 
        # Ensure column names are correct
        df.columns = ["Insert_Name", "TOP", "LEFT", "RIGHT", "BOTTOM"]

        # Append to list
        data_list.append(df)

# Combine all sheets into one DataFrame
final_df = pd.concat(data_list, ignore_index=True)

# Save to CSV
final_df.to_csv(csv_file, index=False)
print(f"CSV file saved as {csv_file}")