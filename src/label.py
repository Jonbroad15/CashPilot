"""
Read in a csv file and label it for categories with user input
"""
import argparse
import os
import pandas as pd
import datetime
import re
from src.model import DataCleaner, SpendingClassifier

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
coffee: coffee, tea, etc.
"""

def clean_data(data_dir):
    for f in os.listdir(data_dir):
        if f.endswith('.csv'):
            os.rename(os.path.join(data_dir, f), os.path.join(data_dir, 'backup', f))

def label(data_dir, master_csv, model_file=None):
    api_key = os.getenv("OPENAI_API_KEY")
    cleaner = DataCleaner(api_key=api_key)
    # get month name in format MMMMM like "January", "February", etc.
    month = datetime.datetime.now().strftime('%B')
    os.makedirs(os.path.join(data_dir, month), exist_ok=True)

    if os.path.exists(master_csv):
        master_df = pd.read_csv(master_csv)
    else:
        print(f"Creating new master CSV file: {master_csv}")
        master_df = pd.DataFrame()
    account_names = set(master_df['account']) if not master_df.empty else set()
    classifier = SpendingClassifier(api_key=api_key, iterations=1000, learning_rate=0.1, depth=6, loss_function='MultiClass')
    if model_file:
        print(f"Loading model from {model_file}")
        classifier.load_model(model_file)
    else:
        print("No model file provided, training a new model.")
        classifier.train(master_df)
        print("Classifier trained on master data.")
        print("Saving model to file.")
        os.makedirs('models', exist_ok=True)
        classifier.save_model(os.path.join('models', f"spending_classifier_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"))


    for file in os.listdir(data_dir):
        if file.endswith(".csv"):
            file_path = os.path.join(data_dir, file)
            print(f"\nProcessing: {file_path}")
            raw_df = pd.read_csv(file_path)
            account_name = file.split('.')[0]

            if account_name not in account_names:
                print(f"New account detected: {account_name}. Adding to master CSV.")
                print("Is this a new account? (y/n)")
                user_input = input().strip().lower()
                if user_input == 'n':
                    raise ValueError(f"Account {account_name} not found in master CSV. Please correct csv naming error and try again.")

            cleaned_df = cleaner.clean(raw_df, account_name=account_name)
            classified_df = classifier.predict(cleaned_df)
            classified_df.to_csv(os.path.join(data_dir, month, file.replace('.csv', '_classified.csv')))

            master_df = pd.concat([master_df, classified_df], ignore_index=True)
            print(f"✔ {file} labeled and added to master.")
            # save master_df
            master_df.drop_duplicates(inplace=True, subset=['date', 'description', 'debit', 'credit'])
            master_df.sort_values(by='date', inplace=True)
            master_df.to_csv(master_csv, index=False)
            # clean up
            os.rename(file_path, os.path.join(data_dir, month, file))


    master_df.to_csv(master_csv, index=False)
    print("\n✅ Master transactions file updated.")
    return master_df

def analyse(df):
    df.drop_duplicates(inplace=True, subset=['date', 'description', 'debit', 'credit'])
    df = df.fillna(0)
    df = df[df['category'] != 'transfers']
    df['date'] = pd.to_datetime(df['date'])
    df['credit'] = df['credit'].astype(float)
    df['debit'] = df['debit'].astype(float)
    df['amount'] = df['credit'] - df['debit']
    df['month_year'] = df['date'].dt.to_period('M')

    result = df.groupby(['month_year', 'category'])['amount'].sum().reset_index()
    summary = result.pivot(index='category', columns='month_year', values='amount').fillna(0)

    category_totals = df.groupby('month_year')[['amount']].sum().reset_index()
    spending = summary.apply(lambda x: x[x < 0].sum(), axis=0)
    income = summary.apply(lambda x: x[x > 0].sum(), axis=0)
    category_totals['spending'] = list(spending)
    category_totals['income'] = list(income)

    summary = summary.round(2)
    print(summary.iloc[:, ::-1])
    print(category_totals)

    return df

def main(args):
    master_df = label(args.data, args.master_csv, model_file=args.model)
    clean_data(args.data)
    print("Data labeled and cleaned. Check the master CSV for potential mistakes.")
    analyse(master_df)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process CSV files and update a master CSV file.")
    parser.add_argument("--master_csv", type=str, default="master.csv", help="Path to the master CSV file")
    parser.add_argument("--data", default='data', help="Path to the data folder")
    parser.add_argument("--model", type=str, default=None, help="Path to the model file for loading or training")
    args = parser.parse_args()
    main(args)
