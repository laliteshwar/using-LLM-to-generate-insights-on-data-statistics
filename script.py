import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import os
import google.generativeai as genai

load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def calc_statistics(df):

    numeric_df = df.select_dtypes(include=['number'])
    stats = {
        'mean': numeric_df.mean(),
        'median': numeric_df.median(),
        'mode': numeric_df.mode().iloc[0],
        'std_dev': numeric_df.std(),
        'correlation': numeric_df.corr()
    }
    return stats

def gen_plots(df):
    numeric_df = df.select_dtypes(include=['number'])
    numeric_df.hist()
    plt.show()

    if numeric_df.shape[1] >= 2:
        plt.scatter(numeric_df.iloc[:, 0], numeric_df.iloc[:, 1])
        plt.xlabel(numeric_df.columns[0])
        plt.ylabel(numeric_df.columns[1])
        plt.show()

    if 'Date' in df.columns and df['Date'].dtype == 'object':
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        df.plot()
        plt.show()

def main():

    df = pd.read_csv('data.csv')

    print("statistics:", calc_statistics(df))
    gen_plots(df)

    stats = calc_statistics(df)
    prompt =(
        "Use the statistics provided to create insights and observations:\n"
        f"mean:{stats['mean']}\n"
        f"median:{stats['median']}\n"
        f"mode:{stats['mode']}\n"
        f"standard Deviation:{stats['std_dev']}\n"
        f"correlation:{stats['correlation']}\n"
    )
    model = genai.GenerativeModel('gemini-1.5-flash')  
    response = model.generate_content(prompt)  
    generated_text = response.text.strip()  

    print("\nGenerated Insights from Google Gemini API:")
    print(generated_text)
    
main()
