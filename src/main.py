import streamlit as st
from scipy.stats import binom
import numpy as np
import pandas as pd

st.set_page_config(layout="wide")

def gen_5star_probs():
    v0 = np.full(72, 0.006)
    v1 = np.linspace(0.006, 1.00, 90-len(v0))
    v = np.concatenate((v0, v1))
    return v

def main():
    # n = 70
    # p = 0.006
    # P = 1 - (1 - p) ** n
    # P = 1 - binom.pmf(k=0, n=n, p=p)
    # print(P)
    # p_increase = (1-0.006)/17
    # v = gen_5star_probs()
    # print(v)
    # print(v[73:])
    # print(len(v))
    # df_p = pd.DataFrame(v[73:], columns=['p'])
    # df_p['p(%)'] = df_p['p'].apply(lambda x: f"{x * 100:.2f}%")
    # print(df_p)
    pass


def gui():
    st.title("Honkai: Star Rail, Probability & Pull Simulator")
    
    p_increase = (1-0.006)/17
    
    v = gen_5star_probs()
    df_p = pd.DataFrame(v[72:], columns=['p'])
    df_show = df_p.copy()
    df_show['p'] = df_show['p'].apply(lambda x: f"{x * 100:.1f}%")
    df_show = df_show.rename(columns={'p': '5⭐ Probability (%)'})
    
    df_show = df_show.T
    df_show.columns = [f'1-73'] + [f'{i+73}' for i in range(1, df_show.shape[1])]

    
    st.write(f'## 5⭐ Drop Probabilities (%)')
    f"""
    The 5⭐ pull probabilities are 0.6% for the first 73 pulls.\n
    The pity system starts with pull 74 which means the probability to get a 5⭐ increases
    by {p_increase*100:.2f}% with every pull until it's {100}% guaranteed on pull 90.\n
    Here in tabular view:
    """
    st.dataframe(df_show, hide_index=True)
    
   

if __name__ in {"__main__", "__mp_main__"}:
    # main()
    gui()

