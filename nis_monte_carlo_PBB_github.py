# Monte Carlo simulation for the German word formation pattern -ieren
# The script randomly re-samples the corpus up to 4m words per decade ('target size')
# For each decade in each corpus version, the script determines a) the number of -nis types
# and b) the number of new -nis types
# The fraction of new -nis types and all -nis types ('pneo') is stored as a csv.

# import necessary libraries
import pandas as pd
import numpy as np
import re
from itertools import compress
from timeit import default_timer as timer
from monte_carlo import MonteCarloSimulation

### set parameters
NUM_PROCESSES = 1
nr_sim = 20                        # nr of simulations
decs = range(1490,2000,10)        # decades in the corpus
decs_result = range(1700,2000,10) # decades of interest
target_size = 4000000             # maximum corpus size per decade

texts_dta = None
suffix_raw = None
tokens_tot = None

# the following function takes a decade (int) as input and returns a list of files similar to texts_dta...
# ...if the real size in tokens exceeds the target size.
# if not, it returns the actual corpus
def shuffle_dec(decade, n):
    real_size = int(tokens_tot.loc[tokens_tot.index == decade]["Freq"])
    if real_size <= target_size:
        return (texts_dta.loc[texts_dta.Dekade == decade])
    else:
        counter = 0
        texts_dec = texts_dta.loc[texts_dta.Dekade == decade].sample(frac = 1)
        j = 1

        while counter < target_size:
            counter += texts_dec.iloc[j-1,2]
            j += 1
        print(decade, counter)
        return (texts_dec.iloc[0:j,:])

# the following function returns (for a given decade) the number of tokens
def find_tokens(df, dec):
    if dec in df.columns.values:
        return(np.sum(df[dec]))
    else:
        return 0

# the following function returns the number words only attested once, the hapax legomena
def find_hapax(val):
    return val[val == 1].shape[0]

# the following function returns (for a given decade) the number of new words that are also hapax legomena
def find_hapax_new(df, dec):
    if dec in df.columns.values:
        return(df[(df["min"] == dec) & (df[dec] == 1)].shape[0])
    else:
        return 0

# the following function finds the earliest attestation (the leftmost non-zero value in a series)
def find_min(series):
    mask = series > 0
    return(series.loc[mask].index[0])

# the following function returns (for a given decade) the number of lemmas that are not zero
def find_types(df, dec):
    if dec in df.columns.values:
        return np.nansum(df[dec] > 0)
    else:
        return 0

def get_suffix_sub(dec, n):
    # create a re-shuffled sub-corpus
    sub_corpus = shuffle_dec(dec, n) # pd.concat([shuffle_dec(dec, n) for dec in decs], axis=0)
    # create a new list of tokens that only includes texts that are in the new re-shuffled version
    suffix_sub = suffix_raw[suffix_raw['Datei'].isin(sub_corpus['Datei'])]
    suffix_sub = suffix_sub.loc[:,['Dekade', 'Lemma']]
    c = sub_corpus['Freq'].sum()
    return suffix_sub

def run_cycle(n):
    suffix_sub = pd.concat([get_suffix_sub(dec, n) for dec in decs], axis=0)
    suffix_sub["Wert"] = 1

    # create a table in which the rows are lexemes and the columns are decades
    suffix_x = pd.pivot_table(suffix_sub, index = 'Lemma', columns = 'Dekade', aggfunc = 'count', fill_value=0)
    suffix_x.columns = list(suffix_x.columns.levels[1])

    # annotate for first occurrence: for each row (=lemma), find the decade of its first occurrence
    suffix_x["min"] = suffix_x.apply(find_min, axis = 1)

    # construct a new table with the number of 'min'-values for each decade of interest (how many -nis-words were new in each decade?)
    new_types_suffix = suffix_x["min"][suffix_x["min"] > 1690].value_counts().to_frame()

    # construct a new dataframe with the decades without new types
    filt = [dec not in new_types_suffix.index for dec in decs_result]
    ind = list(compress(list(decs_result), filt))
    values = [0] * len(ind)

    # merge this new dataframe with the existing one
    new_types_suffix = pd.concat([new_types_suffix, pd.DataFrame(values, index = ind, columns = ["min"])])
    # get the number of all types for each decade
    new_types_suffix["all_types"] = [find_types(suffix_x, dec) for dec in new_types_suffix.index]

    # get the number of all tokens for each decade
    new_types_suffix["all_tokens"] = [find_tokens(suffix_x, dec) for dec in new_types_suffix.index]

    # get the number of all types that only occur in one decade
    new_types_suffix["hapax"] = [find_hapax(suffix_x[dec]) for dec in new_types_suffix.index]

    # get the number of all types that only occur in one decade AND that are new
    new_types_suffix["hapax_new"] = [find_hapax_new(suffix_x, dec) for dec in new_types_suffix.index]

    # Pneo is simply the ratio of all types and new types
    new_types_suffix["pneo"] = new_types_suffix["min"]/new_types_suffix["all_types"]

    return(n, new_types_suffix)

def do_result(result):
    n = result[0]
    print("Simulation %s" % n)
    new_types_suffix = result[1]
    # for each value in the table: set the according value of the global dataframe
    for col, value in new_types_suffix["pneo"].iteritems():
        pneo_global.at[n, col] = value
    for col, value in new_types_suffix["all_tokens"].iteritems():
        tokens_global.at[n, col] = value
    for col, value in new_types_suffix["hapax"].iteritems():
        hapax_global.at[n, col] = value
    for col, value in new_types_suffix["hapax_new"].iteritems():
        hapax_and_new_global.at[n, col] = value
    for col, value in new_types_suffix["min"].iteritems():
        vneo_global.at[n, col] = value

def init_worker(_texts_dta, _suffix_raw, _tokens_tot):
    global texts_dta, suffix_raw, tokens_tot
    texts_dta = _texts_dta
    suffix_raw = _suffix_raw
    tokens_tot = _tokens_tot

if __name__ == "__main__":
    # start the timer
    start = timer()

    # read the necessary files:
    texts_dta  = pd.read_csv('texts_dta_dwds_5000.csv', sep= ',') # this file contains a line for each text in the corpus, together with the respective decade and the token count
    suffix_raw  = pd.read_csv('nis_re-lemmatized_total.csv', sep = ",")       # this file contains a line for each -nis attestation in the corpus, together with the respective decade and filename

    # from the metadata (texts_dta): extract a table with the total amount of tokens per decade
    tokens_tot = pd.pivot_table(texts_dta, index = "Dekade", values = "Freq", aggfunc=np.sum)

    # initialize lists
    pneo_global = pd.DataFrame(columns = decs_result)
    types_global = pd.DataFrame(columns = decs_result)
    tokens_global = pd.DataFrame(columns = decs_result)
    dead_global = pd.DataFrame(columns = decs_result)
    hapax_global = pd.DataFrame(columns = decs_result)
    hapax_and_new_global = pd.DataFrame(columns = decs_result)
    vneo_global = pd.DataFrame(columns = decs_result)

    print("initialization time elapsed: ", timer() - start)
    # restart the timer
    start = timer()
    # this is the actual Monte Carlo simulation, with a for-loop running nr_sim times
    simulation = MonteCarloSimulation(num_processes=NUM_PROCESSES, initializer=init_worker, initargs=(texts_dta, suffix_raw, tokens_tot))
    simulation.start(nr_sim, run_cycle, do_result)

    # save dataframe
    pneo_global.to_csv("NIS_pneo_global_100.csv", encoding = "utf-8")
    tokens_global.to_csv("NIS_tokens_global_100.csv", encoding = "utf-8")
    hapax_global.to_csv("NIS_hapax_global_100.csv", encoding = "utf-8")
    hapax_and_new_global.to_csv("NIS_hapax_and_new_global_100.csv", encoding = "utf-8")
    vneo_global.to_csv("NIS_vneo_global_100.csv", encoding = "utf-8")

    print("time elapsed: ", timer() - start)
