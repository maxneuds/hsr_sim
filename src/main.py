import streamlit as st
from scipy.stats import binom
import numpy as np
import pandas as pd

st.set_page_config(layout="wide")

def gen_p_char():
    v0 = np.full(72, 0.006)
    v1 = np.linspace(0.006, 1.00, 90-len(v0))
    v = np.concatenate((v0, v1))
    return v

def gen_p_lc():
    v0 = np.full(64, 0.008)
    v1 = np.linspace(0.006, 1.00, 80-len(v0))
    v = np.concatenate((v0, v1))
    return v

def p_bernoulli(p, n):
  return 1 - (1 - p) ** n

def gen_cum_drop_chances(v):
    cum_drop_chances = []
    for i, p in enumerate(v):
        if i == 0:
            p = v[i]
        else:
            p = 1 - (1 - cum_drop_chances[i-1]) * (1-v[i])
        cum_drop_chances.append(p)
    return cum_drop_chances

def find_success_index(v):
    # Find the index of the first True value
    index = next((i for i, val in enumerate(v) if val), False)
    return index

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
    
    v = gen_p_char()
    df_p = pd.DataFrame(v[72:], columns=['p'])
    df_show = df_p.copy()
    df_show['p'] = df_show['p'].apply(lambda x: f"{x * 100:.1f}%")
    df_show = df_show.rename(columns={'p': '5⭐ Probability (%)'})
    
    df_show = df_show.T
    df_show.columns = [f'1-73'] + [f'{i+73}' for i in range(1, df_show.shape[1])]


    st.write(f'## Important Simulation results')
    
    f"""
    Mean amount of pulls for limited 5⭐ Character: `{107.35}`\n
    Mean amount of pulls for limited 5⭐ Lightcone: `{73.6375}`\n
    """
    
    st.write(f'## 5⭐ Character Probabilities (%)')
    f"""
    The 5⭐ pull probabilities are 0.6% for the first 73 pulls.\n
    The pity system starts with pull 74 which means the probability to get a 5⭐ increases
    by {p_increase*100:.2f}% with every pull until it's {100}% guaranteed on pull 90.\n
    Here in tabular view:
    """
    st.dataframe(df_show, hide_index=True)
    
    st.write(f'## Total 5⭐ Character Probability (%)')
    f"""
    Given the number of pulls, see the total probability of getting a 5⭐ character.
    """
    p_total = gen_cum_drop_chances(v)
    df_cum = pd.DataFrame({'n': range(1, len(p_total) + 1), 'p': p_total})
    df_cum_show = df_cum.copy()
    df_cum_show['p'] = df_cum['p'].apply(lambda x: f"{x * 100:.2f}%")
    df_cum_show.index = df_cum_show.index + 1
    df_cum_show = df_cum_show[df_cum_show['n'].isin([1, 10, 20, 30, 40, 50, 60, 70]) | (df_cum_show['n'] >= 73)]
    df_cum_show = df_cum_show.rename(columns={'n': 'Number of pulls', 'p': 'Total 5⭐ Probability (%)'})
    n_rows = df_cum_show.shape[0]//2
    df_cum_show_1 = df_cum_show.iloc[:n_rows]
    df_cum_show_2 = df_cum_show.iloc[n_rows+1:]
    row_pcum = st.columns(2)
    with row_pcum[0]:
        st.dataframe(df_cum_show_1, hide_index=True, height=(n_rows+1)*35+3)
    with row_pcum[1]:
        st.dataframe(df_cum_show_2, hide_index=True, height=(n_rows+1)*35+3)

    
    st.write(f'## Simulator: Pull Calculator')
    # setup simulation
    row_pullcalc1 = st.columns(5)
    with row_pullcalc1[0]:
        pullcalc_guranteed = st.selectbox(
            "Guranteed?",
            (False, True),
            index=0,
        )
    with row_pullcalc1[1]:
        n_pullcalc_pity = st.number_input('Pity:', value=0)
    with row_pullcalc1[2]:
        n_pullcalc_limited_chars = st.number_input('Limited Chars:', value=0)
    with row_pullcalc1[3]:
        n_pullcalc_limited_lcs = st.number_input('Limited LCs:', value=0)
    with row_pullcalc1[4]:
        btn_pullcalc_run = st.button("Run", type="primary", key='btn_pullcalc_run')
    # run simulation
    if btn_pullcalc_run:
        for idx_sim in range(10**5):
            True
            # continue here
    
    
    st.write(f'## Simulator: Limited 5⭐ Average Pull Count')
    # setup simulation
    row_limsim_input = st.columns(3)
    with row_limsim_input[0]:
        pull_type_limsim = st.selectbox(
            "Banner Type",
            ("Character", "Lightcone"),
            index=0,
            key = 'pull_type_limsim'
        )
    with row_limsim_input[1]:
        n_sim_limsim = st.number_input('Number of simulations:', value=10**3, key='n_sim_limsim')
    with row_limsim_input[2]:
        btn_run_limsim = st.button("Run", type="primary", key='btn_run_limsim')
    # run simulation
    if btn_run_limsim:
        if pull_type_limsim == 'Lightcone':
            p = gen_p_lc()
            p_limited = 0.75
        else:
            p = gen_p_char()
            p_limited = 0.5
        # simulation runs
        result = []
        for idx_sim in range(n_sim_limsim):
            # pull on gacha simulation
            guranteed = False
            n_plulls = 0
            n_limited = 0
            while n_limited < 1:
                rolls = np.random.rand(len(p))
                drops = rolls <= p
                idx = find_success_index(drops)
                # check if 5⭐ dropped
                n_until_drop = idx + 1
                win5050 = np.random.rand() < p_limited
                if not guranteed and not win5050:
                    guranteed = True
                    n_plulls += len(drops)
                else:
                    n_limited += 1
                    guranteed = False
                    n_plulls += n_until_drop
            # add simulation result to list of results
            result_step = {'idx_sim': idx_sim+1, 'n_pulls': n_plulls}
            result.append(result_step)
        # stats
        df = pd.DataFrame(result)
        mean_pulls = df['n_pulls'].mean()
        st.write(f'Mean amout of pulls for limited 5⭐: {mean_pulls}')


    st.write(f'## Simulator: 5⭐ Count From Pulls')
    # setup simulation
    row_pullsim_input = st.columns(4)
    with row_pullsim_input[0]:
        n_plulls = st.number_input('Number of pulls:', value=100)
    with row_pullsim_input[1]:
        pull_type_pullsim = st.selectbox(
            "Banner Type",
            ("Character", "Lightcone"),
            index=0,
            key = 'pull_type_pullsim'
        )
    with row_pullsim_input[2]:
        n_sim_pullsim = st.number_input('Number of simulations:', value=10**3, key='n_sim_pullsim')
    with row_pullsim_input[3]:
        btn_run_pullsim = st.button("Run", type="primary", key='btn_run_pullsim')
    # run simulation
    if btn_run_pullsim:
        if pull_type_pullsim == 'Lightcone':
            p = gen_p_lc()
            p_limited = 0.75
        else:
            p = gen_p_char()
            p_limited = 0.5
        # simulation runs
        result = []
        for idx_sim in range(n_sim_pullsim):
            # pull on gacha simulation
            guranteed = False
            n_standard = 0
            n_limited = 0
            n_rest = n_plulls
            len_p = len(p)
            while n_rest > 0:
                if n_rest >= len_p:
                    n = min(n_rest, len_p)
                    p_sim = p
                else:
                    n = n_rest
                    p_sim = p[:n]
                rolls = np.random.rand(n)
                drops = rolls <= p_sim
                idx = find_success_index(drops)
                # check if 5⭐ dropped, and if yes check for limited
                if idx != False:
                    n_until_drop = idx + 1
                    win5050 = np.random.rand() < p_limited
                    if not guranteed and not win5050:
                        n_standard += 1
                        guranteed = True
                    else:
                        n_limited += 1
                        guranteed = False
                else:
                    n_until_drop = n
                # continue if still pulls available
                n_rest = n_rest - n_until_drop
            # add simulation result to list of results
            result_step = {'idx_sim': idx_sim+1, 'n_limited': n_limited, 'n_standard': n_standard}
            result.append(result_step)
        # stats
        df = pd.DataFrame(result)
        df['n_total'] = df['n_limited'] + df['n_standard']
        n_limited_mean = df['n_limited'].mean()
        n_standard_mean = df['n_standard'].mean()
        n_total_mean = df['n_total'].mean()
        st.write(f'Mean amount of limited 5⭐: {n_limited_mean}')
        st.write(f'Mean amount of standard 5⭐: {n_standard_mean}')
        st.write(f'Mean amount of total 5⭐: {n_total_mean}')


if __name__ in {"__main__", "__mp_main__"}:
    # main()
    gui()

