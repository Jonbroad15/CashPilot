import argparse
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

def main(args):
    
    df = pd.read_csv(args.master)
    df.drop_duplicates(inplace=True)
    df = df.fillna(0)
    df['date'] = pd.to_datetime(df['date'])
    df['amount'] = df['credit'] - df['debit']
    df['month_year'] = df['date'].dt.to_period('M')
    result = df.groupby(['month_year', 'category'])['amount'].sum().reset_index()

    breakpoint()

    # Set a style for the plots (optional)
    sns.set_style('whitegrid')

    # Create a bar plot
    plt.figure(figsize=(12, 6))
    sns.barplot(x='month_year', y='amount', hue='category', data=result)
    plt.title('Monthly Spending by Category')
    plt.xlabel('Month-Year')
    plt.ylabel('Total Amount')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--master', default = 'master.csv')
    args = p.parse_args()
    main(args)
