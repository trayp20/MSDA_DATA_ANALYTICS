import pandas as pd

# explicitly load your formatted data
df = pd.read_csv('cleaned_data.csv')

# Filter explicitly for ATL airport
df_filtered_airport = df[df['ORG_AIRPORT'] == 'ATL']

# explicitly export the filtered dataset
df_filtered_airport.to_csv('cleaned_data_airport.csv', index=False)

# Explicitly remove rows with missing delay values
df = df.dropna(subset=['DEPARTURE_DELAY', 'ARRIVAL_DELAY'])

# Explicitly remove outliers (>60 mins delays)
df = df[(df['DEPARTURE_DELAY'] <= 60) & (df['ARRIVAL_DELAY'] <= 60)]

# Reset index clearly
df = df.reset_index(drop=True)

# explicitly save the fully cleaned data
df.to_csv('cleaned_data.csv', index=False)