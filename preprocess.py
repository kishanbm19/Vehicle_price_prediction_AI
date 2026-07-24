import pandas as pd

def pre_process(df):
    df=df.dropna()
    def convert(price):
        price=price.replace('₹',"").strip()
        if 'Lakh' in price:
            return float(price.replace('Lakh',"").strip())*100000
        elif 'Crore' in price:
            return float(price.replace('Crore',"").strip())*10000000
        return None

    df['car_price_in_rupees']=df['car_price_in_rupees'].apply(convert)
    df = df.dropna(subset=["car_price_in_rupees"])
    df['kms_driven']=(
        df['kms_driven']
        .str.replace(',',"",regex=False)
        .str.replace('km',"",regex=False)
        .str.strip()
        .astype(int)
    )

    df['brand']=df['car_name'].str.split().str[0]
    current_yr=2026

    df['car_age']=current_yr-df['year_of_manufacture']

    df=df.drop(["car_name","year_of_manufacture"],axis=1)

    
    return df







