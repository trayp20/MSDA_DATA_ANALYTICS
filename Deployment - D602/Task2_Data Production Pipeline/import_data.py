import pandas as pd


df = pd.read_csv('/Users/trayvoniouspendleton/IdeaProjects/d602-deployment-task-2/On_Time_Reporting_Carrier_On_Time_Performance_(1987_present)_2024_12.csv', low_memory=False)

# Filter relevant columns
selected_columns = ['Year','Month','DayofMonth','DayOfWeek','Origin', 'Dest', 'CRSDepTime', 'DepTime', 'DepDelay', 'CRSArrTime', 'ArrTime', 'ArrDelay']

df_filtered = df[selected_columns]

# Save formatted data
df_filtered.to_csv('formatted_airline_data.csv', index=False)

# Rename columns according to poly_regressor_Python script requirement
df_formatted = df.rename(columns={
    'DayofMonth': 'DAY',
    'Origin': 'ORG_AIRPORT',
    'CRSDepTime': 'SCHEDULED_DEPARTURE',
    'DepTime': 'DEPARTURE_TIME',
    'DepDelay': 'DEPARTURE_DELAY',
    'CRSArrTime': 'SCHEDULED_ARRIVAL',
    'ArrTime': 'ARRIVAL_TIME',
    'ArrDelay': 'ARRIVAL_DELAY',
    'Dest': 'DEST_AIRPORT',
    'Year':'YEAR',
    'Month': 'MONTH',
    'DayOfWeek': 'DAY_OF_WEEK'


})

# Select explicitly required columns only
columns_required = [
    'YEAR', 'MONTH', 'DAY', 'DAY_OF_WEEK', 'ORG_AIRPORT', 'DEST_AIRPORT',
    'SCHEDULED_DEPARTURE', 'DEPARTURE_TIME', 'DEPARTURE_DELAY',
    'SCHEDULED_ARRIVAL', 'ARRIVAL_TIME', 'ARRIVAL_DELAY'
]

df_cleaned = df_formatted[columns_required]

# Save formatted CSV
df_formatted.to_csv('cleaned_data.csv', index=False)

