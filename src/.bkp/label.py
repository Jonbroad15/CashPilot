"""
Read in a csv file and label it for catagories with user input
"""
import argparse
import os
import pandas as pd
import xlrd
import datetime
import re

USD_TO_CAD_RATE = 1.345

category_descriptions = """
housing: Rent or mortgage payments, property taxes, and home insurance.
utilities: Electricity, water, gas, internet bills, phone bills.
transit: Car payments, fuel, public transportation, and maintenance.
groceries: Food and household essentials.
savings: Emergency fund contributions, retirement savings, and other investments.
entertainment: Dining out, movies, concerts, and leisure activities.
wellness: Gym memberships, uncovered healthcare, cosmetic and hygiene products
gifts: Presents for birthdays, holidays, and charitable donations.
restaurants: Eating out at restaurants and bars
transfers: Transfer to other accounts
fees: Banking fees (interest)
supplies: canadian tire, home improvement, computers, electronics
clothing: buying new clothes
purchases: large purchases for myself
investments: Money loss or gained from investments
income: money gained from employment income
travel: travel expenses
rebates:
cash:
catpital_gl: Capital gains/losses
alcohol: alcohol, weed, etc.
coffee: coffe, tea, etc.
"""
def clean_data(data_dir):
    os.makedirs(os.path.join(data_dir, 'backup'), exist_ok=True)
    for f in os.listdir(data_dir):
        # move all csv files to backup folder
        if f.endswith('.csv'):
            os.rename(os.path.join(data_dir, f), os.path.join(data_dir, 'backup', f))
    return

def label(data, master_csv):
    """
    reads all csv files in data and labels them into master_csv

    :param data: path ot directory containing bank statement csv files
    :param master_csv: path to master_csv file containing all labelled bank statements
    """
    # if there is a wealthsimple.txt file, convert it to csv

    # Get the list of CSV files in the 'data' folder

    print("Master transactions file updated.")

    return master_df

def analyse(df):

    df.drop_duplicates(inplace=True, subset=['date', 'description', 'debit', 'credit'])
    df = df.fillna(0)
    df['date'] = pd.to_datetime(df['date'])
    df['credit'] = df['credit'].astype(float)
    df['debit'] = df['debit'].astype(float)
    df['amount'] = df['credit'] - df['debit']
    df['month_year'] = df['date'].dt.to_period('M')

    result = df.groupby(['month_year', 'category'])['amount'].sum().reset_index()
    summary = result.pivot(index='category', columns='month_year', values='amount').fillna(0)

    category_totals = df.groupby('month_year')[['amount']].sum().reset_index()
    # make 'spending' and 'income' columns on the summary dataframe
    # spending is the sum of all negative values
    # income is the sum of all positive values
    spending = summary.apply(lambda x: x[x < 0].sum(), axis=0)
    income = summary.apply(lambda x: x[x > 0].sum(), axis=0)
    category_totals['spending'] = list(spending)
    category_totals['income'] = list(income)

    # round values to 2 decimal places
    summary = summary.round(2)
    # print summary in reverse column order
    print(summary.iloc[:, ::-1])
    print(category_totals)

    return df


def main(args):
    master_df = label(args.data, args.master_csv)
    clean_data(args.data)
    if args.accounts:
        tfsa_contribution()
    else:
        df = pd.read_csv('accounts.csv', index_col=0, parse_dates=True)
        for col in df.columns:
            print(f"{col}: \t\t\t {df[col].iloc[-1]:.2f} \t\t {df[col].iloc[-1] - df[col].iloc[-2]:.2f} \t\t {df[col].iloc[-2]:.2f}")
    df = analyse(master_df)

    # Compute TFSA Contribution Room

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process CSV files and update a master CSV file.")
    parser.add_argument("--master_csv", type=str, default="master.csv", help="Path to the master CSV file")
    parser.add_argument("--data", default=None, help="Path to the data folder")
    parser.add_argument("--accounts", type=bool, default=False, help="Update accounts file")
    args = parser.parse_args()
    main(args)

