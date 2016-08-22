from sqlalchemy import create_engine
import numpy as np
import pandas as pd

engine = create_engine('postgresql://kyle.joecken@localhost:5432/nhlstats')

df = pd.read_sql_query('select * from skaterstats where season = 2013', con=engine)
df.drop('id', 1, inplace=True)
df.drop('season', 1, inplace=True)

if __name__ == "__main__":
    print(df.head())
