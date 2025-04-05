
import pandas as pd
from datetime import datetime
# Load all raw files 

# KEYS.csv
keys_df = pd.read_csv("KEYS.csv")

# CENSUS_MHI_STATE.csv
mhi_df = pd.read_csv("CENSUS_MHI_STATE.csv")

# CENSUS_POPULATION_STATE.tsv
population_df = pd.read_csv("CENSUS_POPULATION_STATE.tsv", sep="\t")

# REDFIN_MEDIAN_SALE_PRICE.csv
redfin_df = pd.read_csv("REDFIN_MEDIAN_SALE_PRICE.csv")
# Step 2 : Setup a lookup column of States
# Column A → key_row (keys_df.iloc[:, 0])
# Column F → region_type or label (keys_df.iloc[:, 5])

df_result = keys_df[
    (keys_df.iloc[:, 5] == 'state') &
    (~keys_df.iloc[:, 0].str.contains("'", na=False))
].copy()

# Reset index and keep only key_row and census_msa (needed for lookups later)
df_result = df_result.reset_index(drop=True)
df_result = df_result[['key_row', 'census_msa']]

# review result
print(df_result.head())

# Step 3 : Define & Apply a function for population lookup  
def get_census_population_for_key_row(key_row):
    match_row = keys_df[keys_df['key_row'] == key_row]
    if match_row.empty:
        return None
    census_msa = match_row.iloc[0]['census_msa']
    col_name = f"{census_msa}!!Estimate"

    pop_row = population_df[population_df[population_df.columns[0]].str.contains("Total population", na=False)]
    if pop_row.empty or col_name not in population_df.columns:
        return None

    value = pop_row.iloc[0][col_name]
    if isinstance(value, str):
        value = value.replace(',', '').strip()
    try:
        return int(value)
    except:
        return None

# Apply the lookup for each key_row
df_result['census_population'] = df_result['key_row'].apply(get_census_population_for_key_row)

# review result
print(df_result.head())

# Step 4: Rank and format population 
def ordinal_suffix(rank):
    if pd.isnull(rank):
        return ""
    rank = int(rank) 
    if 11 <= (rank % 100) <= 13:
        return f"{rank}th"
    else:
        return f"{rank}{['th', 'st', 'nd', 'rd', 'th', 'th', 'th', 'th', 'th', 'th'][rank % 10]}"



# Compute population rank (1 = highest)
df_result['population_rank'] = df_result['census_population'].rank(ascending=False, method='min').astype(int)

# Format as ordinal string
df_result['population_rank'] = df_result['population_rank'].apply(ordinal_suffix)

# Review result
print(df_result.head())

# Step 5: Create population_blurb using KEYS column H (alternative_name)
# Load mapping of key_row -> alternative_name
key_row_to_alt_name = keys_df.set_index('key_row')['alternative_name'].to_dict()

df_result['population_blurb'] = df_result.apply(
    lambda row: f"{key_row_to_alt_name.get(row['key_row'], '')} is {row['population_rank']} in the nation in population among states, DC, and Puerto Rico.",
    axis=1
)

# Review result
print(df_result.head())

# Step 6: Lookup median_household_income 
def get_income_for_key_row(key_row):
    match_row = keys_df[keys_df['key_row'] == key_row]
    if match_row.empty:
        return None
    census_msa = match_row.iloc[0]['census_msa']
    col_name = f"{census_msa}!!Median income (dollars)!!Estimate"

    income_row = mhi_df[mhi_df[mhi_df.columns[0]].str.contains("Households", na=False)]
    if income_row.empty or col_name not in mhi_df.columns:
        return None

    value = income_row.iloc[0][col_name]
    if isinstance(value, str):
        value = value.replace(',', '').replace('±', '').strip()
    try:
        return int(value)
    except:
        return None

df_result['median_household_income'] = df_result['key_row'].apply(get_income_for_key_row)

# Review result
print(df_result.head())

# Step 7: Rank Income & Add Ordinal Label
df_result['median_household_income_rank'] = df_result['median_household_income'].rank(ascending=False, method='min').astype(int)
df_result['median_household_income_rank'] = df_result['median_household_income_rank'].apply(ordinal_suffix)

# Review result
print(df_result.head())

# Step 8: Income blurb like:

df_result['median_household_income_blurb'] = df_result.apply(
    lambda row: (
        f"{key_row_to_alt_name.get(row['key_row'], '')} is "
        f"{'the highest' if row['median_household_income_rank'] == '1st' else row['median_household_income_rank']} "
        "in the nation in median household income among states, DC, and Puerto Rico."
    ),
    axis=1
)

# Review result
print(df_result.head())

# Step 9: Redfin Median Sale Price (latest month)

# Prepare Redfin dataset
redfin_data = redfin_df.copy()
redfin_data.columns = redfin_data.iloc[0]  # Set first row as header
redfin_data = redfin_data.iloc[1:]         # Remove header row

# Get the latest available month (last column)
price_columns = [col for col in redfin_data.columns if col and 'Region' not in str(col)]
latest_month_col = price_columns[-1]

# Clean and normalize
redfin_data['state_key'] = redfin_data['Region'].str.strip().str.lower()
redfin_data['latest_median_sale_price'] = (
    redfin_data[latest_month_col]
    .replace('[\$,K]', '', regex=True)
    .replace('', None)
    .astype(float) * 1000
)

# Map sale price to each state
state_price_dict = redfin_data.set_index('state_key')['latest_median_sale_price'].to_dict()
df_result['state_key'] = df_result['key_row'].str.replace("_", " ").str.lower()
df_result['median_sale_price'] = df_result['state_key'].map(state_price_dict)

# Review result
print(df_result.head())

# Step 10: Rank and format median sale price
valid_prices = df_result['median_sale_price'].notnull()
df_result.loc[valid_prices, 'median_sale_price_rank'] = (
    df_result.loc[valid_prices, 'median_sale_price']
    .rank(ascending=False, method='min')
)

df_result['median_sale_price_rank_int'] = df_result['median_sale_price_rank'].round().astype('Int64')
df_result['median_sale_price_rank'] = df_result['median_sale_price_rank_int'].apply(
    lambda x: ordinal_suffix(x) if pd.notnull(x) else ''
)

# Review result
print(df_result.head())

# Step 11: Sale Price Blurb 
# Parse Redfin column name as Month Year

try:
    latest_date_obj = pd.to_datetime(latest_month_col, errors='coerce')
    formatted_month_year = latest_date_obj.strftime("%B %Y") if not pd.isnull(latest_date_obj) else latest_month_col
except:
    formatted_month_year = latest_month_col

# Now create the blurb using the rank column
df_result['median_sale_price_blurb'] = df_result.apply(
    lambda row: (
        f"{key_row_to_alt_name.get(row['key_row'], '')} has the "
        f"{'single' if row['median_sale_price_rank'] == '1st' else row['median_sale_price_rank']} "
        f"highest median sale price on homes in the nation among states, DC, and Puerto Rico, "
        f"according to Redfin data from {formatted_month_year}."
    ),
    axis=1
)

# Review result
print(df_result.head())

# Step 12: Calculate house affordability ratio

df_result['house_affordability_ratio'] = (
    df_result['median_sale_price'] / df_result['median_household_income']
).round(1)

# Review result
print(df_result.head())

# Step 13: Rank the house affordability ratio


# Only rank rows with valid affordability ratios
valid_afford = df_result['house_affordability_ratio'].notnull()

# Calculate ascending rank (1 = most affordable)
df_result.loc[valid_afford, 'house_affordability_ratio_rank'] = (
    df_result.loc[valid_afford, 'house_affordability_ratio']
    .rank(ascending=True, method='min')
)

# Keep as Int64 to allow for missing values
df_result['house_affordability_ratio_rank_int'] = (
    df_result['house_affordability_ratio_rank'].round().astype('Int64')
)

df_result['house_affordability_ratio_rank'] = df_result[
    'house_affordability_ratio_rank_int'
].apply(ordinal_suffix)

# Review result
print(df_result.head())

# Step 14: Affordability Ratio Blurb 

# Build the affordability blurb
df_result['house_affordability_ratio_blurb'] = df_result.apply(
    lambda row: (
        f"{key_row_to_alt_name.get(row['key_row'], '')} has the "
        f"{'single' if row['house_affordability_ratio_rank'] == '1st' else row['house_affordability_ratio_rank']} "
        f"lowest house affordability ratio in the nation among states, DC, and Puerto Rico, "
        f"according to Redfin data from {formatted_month_year}."
    ),
    axis=1
)

# Review result
print(df_result.head())
 
# Format census_population with commas (e.g., 5,108,468)
df_result['census_population'] = df_result['census_population'].apply(
    lambda x: f"{int(x):,}" if pd.notnull(x) else ""
)

# Format median_household_income with $ and commas (e.g., $96,334)
df_result['median_household_income'] = df_result['median_household_income'].apply(
    lambda x: f"${int(x):,}" if pd.notnull(x) else ""
)

# Format median_sale_price with $ and commas (e.g., $833,000)
df_result['median_sale_price'] = df_result['median_sale_price'].apply(
    lambda x: f"${int(x):,}" if pd.notnull(x) else ""
)
# Final Export 
final_columns = ['key_row','census_population','population_rank','population_blurb', 'median_household_income', 'median_household_income_rank', 'median_household_income_blurb', 'median_sale_price','median_sale_price_rank', 'median_sale_price_blurb','house_affordability_ratio','house_affordability_ratio_rank','house_affordability_ratio_blurb']
final_csv_df = df_result[final_columns]
final_csv_df.to_csv("final_state_analysis.csv", index=False)
# Optional Visualizations

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the final CSV 
df = pd.read_csv("final_state_analysis.csv")

# Optional: remove $ and commas to convert numeric columns back
df['median_sale_price'] = df['median_sale_price'].replace('[\$,]', '', regex=True).astype(float)
df['median_household_income'] = df['median_household_income'].replace('[\$,]', '', regex=True).astype(float)
df['census_population'] = df['census_population'].replace(',', '', regex=True).astype(int)
df['house_affordability_ratio'] = pd.to_numeric(df['house_affordability_ratio'], errors='coerce')

# 1. Top 10 states by population
top_pop = df.sort_values(by='census_population', ascending=False).head(10)
plt.figure(figsize=(10, 6))
sns.barplot(x='census_population', y='key_row', data=top_pop, palette='Blues_d')
plt.title("Top 10 States by Population")
plt.xlabel("Population")
plt.ylabel("State")
plt.tight_layout()
plt.savefig("top_10_population.png")
plt.close()

# 2. Top 10 states by median sale price
top_price = df.sort_values(by='median_sale_price', ascending=False).head(10)
plt.figure(figsize=(10, 6))
sns.barplot(x='median_sale_price', y='key_row', data=top_price, palette='Reds_d')
plt.title("Top 10 States by Median Sale Price")
plt.xlabel("Median Sale Price ($)")
plt.ylabel("State")
plt.tight_layout()
plt.savefig("top_10_sale_price.png")
plt.close()

# 3. Top 10 most affordable states (lowest affordability ratio)
most_affordable = df.sort_values(by='house_affordability_ratio').head(10)
plt.figure(figsize=(10, 6))
sns.barplot(x='house_affordability_ratio', y='key_row', data=most_affordable, palette='Greens_d')
plt.title("Top 10 Most Affordable States")
plt.xlabel("Affordability Ratio")
plt.ylabel("State")
plt.tight_layout()
plt.savefig("most_affordable_states.png")
plt.close()

# 4. Scatter plot: Income vs. Home Price
plt.figure(figsize=(8, 6))
sns.scatterplot(x='median_household_income', y='median_sale_price', data=df)
plt.title("Median Household Income vs. Median Sale Price")
plt.xlabel("Median Household Income ($)")
plt.ylabel("Median Sale Price ($)")
plt.tight_layout()
plt.savefig("income_vs_sale_price.png")
plt.close()
 