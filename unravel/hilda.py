#!/bin/env python3
#
# hilda.py - for (pre)processing of HILDA data.
#
# Author: Fjalar de Haan (fjalar.dehaan@unimelb.edu.au)
# Created: 2022-09-12
# Last modified: 2023-04-25
#

import pickle, os

import numpy as np
import pandas as pd
import pyreadstat

# Path strings to 20th ('t') wave of HILDA data set.
project_path = ( "/home/fjalar/pCloudDrive/"
                 "archive/academia/projects/future-of-work/" )
hilda_spss_path = "/server/data/hilda/spss-200c/Combined t200c.sav"
hilda_pickle_path = project_path + "data/hilda/hilda-combined-t200c.pickle"
raw_pickle_path = project_path + "data/hilda/hilda-combined-t200c-raw.pickle"

def clean(raw, fill='mode'):
    """Clean HILDA data."""
    # Exclude `object` cols containing wave ids, dates and other irrelevantia.
    cleaned = raw.select_dtypes(include='float64').copy()
    # Drop columns with only NaNs.
    cleaned.dropna(axis='columns', how='all', inplace=True)
    # Replace ramaining NaNs.
    if fill == 'mean':
        # Replace NaNs with mean values --- this messes up variables like `sex`.
        cleaned.fillna(cleaned.mean().to_dict(), inplace=True)
    else:
        # Replace NaNs with most common values.
        d = cleaned.mode().to_dict()
        replacements = {key: val[0] for (key, val) in d.items()}
        cleaned.fillna(replacements, inplace=True)
    # Drop columns with only one value.
    cols = [ col for col in cleaned.columns
                 if pd.unique(cleaned[col]).shape[0]==1 ]
    cleaned.drop(columns=cols, inplace=True)
    return cleaned

def stats(data):
    """Some statistics of the data."""
    rowlabels = [ "nunique"
                , "min"
                , "mean"
                , "max"
                , "std" ]
    df = pd.DataFrame(columns=data.columns, index=rowlabels)
    for col in data.columns:
        df.loc["nunique", col] = pd.unique(data[col]).shape[0]
    df.loc["min"] = data.min()
    df.loc["mean"] = data.mean()
    df.loc["max"] = data.max()
    df.loc["std"] = data.std()
    return df

jcols = { 'tlosatsf': 'Life satisfaction level'
        , 'tjbhruc': 'Combined per week usually worked in all jobs'
        , 'thiwsfei': 'Imputed financial year gross wages and salary'
        , 'tedhists': 'highest level of education'
        , 'thhda10': 'SEIFA decile of socio-economic disadvantage'
        }

fcols = { # Basic demographics.
          'thgage': 'DV Age last birthday at June 30'
        , 'thgsex': 'Sex'
        , 'tmrcurr': 'Marital status'
        , 'tedhigh1': 'Highest education level achieved'
        , 'tedlhqn': 'Highest education level'
        , 'tes': 'Employment  status'
        , 'thhda10': 'SEIFA decile of socio-economic disadvantage'
          # Work-related factors.
        , 'tjbmsall': 'Overall job satisfaction'
        , 'tjbmsflx': 'Flexibility to balance work/life satisfaction'
        , 'tesdtl': 'Labour force status detailed'
        , 'tjbmshrs': 'Hours working satisfaction'
        , 'twscei': 'Current gross weekly income'
        , 'twsfei': 'Financial year gross income'
        , 'tjbmspay': 'Total pay satisfaction'
        , 'tjbmhruc': 'Hours per week in main job'
        , 'tjbhruc': 'Hours per week in all jobs'
        , 'tjbhrcpr': 'Preference to work fewer/same/more hours'
        , 'tjbmssec': 'Job security satisfaction'
        , 'tjbmswrk': 'Work itself satisfaction'
        , 'tlosateo': 'Employment opportunities satisfaction'
        , 'tjbmo62': 'Occupation code 2-digit ANZSCO'
        , 'tjbmcnt': 'Employment contract (current job)'
        , 'tjbmh': 'Any usual working hours worked from home'
        , 'tjbmlha': 'Main job location varies'
        # (Not in index) , 'tjbmlkm': 'Main job location distance from home'
          # Health-related factors.
        , 'tlosatyh': 'Health satisfaction'
        , 'tgh1': 'Self-assessed health'
        , 'tghgh': 'SF-36 general health'
        , 'tghmh': 'SF-36 mental health'
        , 'tjomms': 'Job is more stressful than I had ever imagined'
        , 'tlosat': 'Life satisfaction'
        }

cols = {
       # 'thgage': 'Age (approx.)'
       # , 'thgsex': 'Sex'
       # , 'tmrcurr': 'Marital status'
       # , 'tedhigh1': 'Highest education level achieved'
       # , 'tes': 'Employment  status'
        'thhda10': 'SEIFA decile of socio-economic disadvantage'
       , 'tjbmsall': 'Overall job satisfaction'
       , 'tghmh': 'SF-36 mental health'
       }

bcols = [ 'tjomus'
        , 'tskcjed'
        , 'tjomcd'
        , 'tjomfast'
        , 'tjomms'
        , 'tjompi'
        , 'tjomtime'
        , 'tjomwi'
        , 'tjbempt'
        # , 'tjbmcntr' # Only in Wave 1.
        , 'tjbmploj'
        , 'tjbmssec'
        , 'tjbocct'
        , 'tjomsf'
        , 'tjomwf'
        , 'tlosateo'
        # , 'tjoskill' # Only in Wave 5.
        , 'tjomns'
        , 'tjomls'
        , 'tjomini'
        , 'tjowpcc'
        # , 'tjowpcr' # Only in  1 < Wave < 20.
        , 'tjowpptw'
        # , 'tjowpuml' # Only in 1 < Wave < 11.
        , 'tjowppml'
        # , 'tjowppnl' # Only in 1 < Wave < 11.
        , 'tjompf'
        , 'twscei'
        , 'twscg'
        , 'twsfei'
        , 'twsfes'
        , 'tjbmswrk'
        # , 'tjonomfl' # Only in Wave 5.
        # , 'tjoserve' # Only in Wave 5.
        # , 'tjosoc' # Only in Wave 5.
        # , 'tjosat' # Only in Wave 5.
        , 'tjbmsall'
        # , 'tjostat' # Only in Wave 5.
        # , 'tjonovil' # Only in Wave 5.
        , 'tjomdw'
        , 'tjomrpt'
        , 'tjomvar'
        , 'tjbmagh'
        , 'tjbmh'
        , 'tjbmhl'
        # , 'tjbmhrh' # Only in Wave 1.
        # , 'tjbmhrha' # Not enough rows, drops out after sampling.
        , 'tjbmhruc'
        , 'tjbmsl'
        , 'tjowpfx'
        , 'tjowphbw'
        , 'tjombrk'
        , 'tjomdw'
        , 'tjomfd'
        , 'tjomflex'
        , 'tjomfw'
        # , 'tjbtremp' # Only in 2 <Wave < 7
        , 'tjbmsflx'
        , 'tjbhruc'
        # , 'tjbhru' # Only in Wave 1.
        # , 'tjbmhruw' # Not enough rows, drops out after sampling.
        # , 'tatwkhpj' # Only in Wave 1.
        , 'tlosat'
        , 'tjbmshrs'
        , 'tjbmspay'
        , 'tlosatfs'
        , 'tlosatft'
        , 'tjbnewjs'
        , 'tjompi'
        , 'tlosatyh' ]

contractions = { 'authority': ['tjomls']
               , 'autonomy': ['tjomini']
               , 'career and skill development (growth)': [ 'tjoskill'
                                                          , 'tjomns']
               , 'career opportunities': ['tlosateo']
               , 'flexible work practices': [ 'tjbmagh'
                                            , 'tjbmh'
                                            , 'tjbmhl'
                                            , 'tjbmhrh'
                                            , 'tjbmhrha'
                                            , 'tjbmhruc'
                                            , 'tjbmsl'
                                            , 'tjowpfx'
                                            , 'tjowphbw'
                                            , 'tjombrk'
                                            , 'tjomdw'
                                            , 'tjomfd'
                                            , 'tjomflex'
                                            , 'tjomfw' ]
               , 'income': [ 'tjowppml'
                           , 'tjowppnl'
                           , 'tjompf'
                           , 'twscei'
                           , 'twscg'
                           , 'twsfei'
                           , 'twsfes' ]
               , 'job attitudes': [ 'tjbmshrs'
                                  , 'tjbmspay'
                                  , 'tlosatfs'
                                  , 'tlosatft' ]
               , 'job demand (stress)': [ 'tjomcd'
                                        , 'tjomfast'
                                        , 'tjomms'
                                        , 'tjompi'
                                        , 'tjomtime'
                                        , 'tjomwi' ]
               , 'job resources': [ 'tjowpcc'
                                  , 'tjowpcr'
                                  , 'tjowpptw'
                                  , 'tjowpuml' ]
               , 'job-satisfaction': ['tjosat', 'tjbmsall']
               , 'life satisfaction': ['tatwkhpj', 'tlosat']
               , 'long term employment and job security': [ 'tjbempt'
                                                          , 'tjbmcntr'
                                                          , 'tjbmploj'
                                                          , 'tjbmssec'
                                                          , 'tjbocct'
                                                          , 'tjomsf'
                                                          , 'tjomwf' ]
               , 'recognition': ['tjostat']
               , 'skill-job fit': ['tjomus', 'tskcjed']
               , 'communication with co-workers': ['tjosoc']
               , 'turnover intentions': ['tjbnewjs']
               , 'variety': ['tjonovil', 'tjomdw', 'tjomrpt', 'tjomvar']
               , 'well being (mental and physical)': ['tjompi', 'tlosatyh']
               , 'work engagement': ['tjbmswrk', 'tjonomfl', 'tjoserve']
               , 'work-life balance': ['tjbmsflx']
               , 'working hours': ['tjbhruc', 'tjbhru', 'tjbmhruw']
               , 'workplace training satisfaction': ['tjbtremp']
               }
ISCO88 ={ 11: "Legislators and senior officials"
        , 12: "Corporate managers"
        , 13: "General managers"
        , 21: "Physical, mathematical and engineering science professionals"
        , 22: "Life science and health professionals"
        , 23: "Teaching professionals"
        , 24: "Other professionals"
        , 31: "Physical and engineering science associate professionals"
        , 32: "Life science and health associate professionals"
        , 33: "Teaching associate professionals"
        , 34: "Other associate professionals"
        , 41: "Office clerks"
        , 42: "Customer services clerks"
        , 51: "Personal and protective services workers"
        , 52: "Models, salespersons and demonstrators"
        , 61: "Market-oriented skilled agricultural and fishery workers"
        , 62: "Subsistence agricultural and fishery workers"
        , 71: "Extraction and building trades workers"
        , 72: "Metal, machinery and related trades workers"
        , 73: "Precision, handicraft, craft printing and related trades workers"
        , 74: "Other craft and related trades workers"
        , 81: "Stationary plant and related operators"
        , 82: "Machine operators and assemblers"
        , 83: "Drivers and mobile plant operators"
        , 91: "Sales and services elementary occupations"
        , 92: "Agricultural, fishery and related labourers"
        , 93: "Labourers in mining, construction, manufacturing and transport" }

# Read the HILDA data one way or another.
if os.path.exists(hilda_spss_path):
    raw, meta = pyreadstat.read_sav(hilda_spss_path)
    hilda = clean(raw)
else:
    with open(hilda_pickle_path, "rb") as f:
        hilda, meta = pickle.load(f)
    with open(raw_pickle_path, "rb") as f:
        raw = pickle.load(f)

# Produce subsets of HILDA based on Fjalar, Brandon or Josh's columns.
hildaf = hilda[fcols.keys()]
hildab = hilda[bcols]
hildaj = hilda[jcols.keys()]
hilda1k = clean(raw.sample(n=1000, random_state=999))
hilda100 = clean(raw.sample(n=100, random_state=999))
hilda25 = clean(raw.sample(n=25, random_state=999))
h25x500 = hilda25.sample(n=500, axis='columns', random_state=11)
# Below subset contains 'tjbmsall':
h100x300 = hilda100.sample(n=300, axis='columns', random_state=99)
h100x300_2 = hilda100.sample(n=300, axis='columns', random_state=9999)

# All the ISCO88 codes with rows in HILDA.
iscosraw = [ int(code) for code in hilda['tjbm682'].unique() ]

# Subsetting.
iscos = [ (isco, hilda[hilda['tjbm682'] == isco].shape[0]) for isco in iscosraw
          if isco not in [1, 34] # Dubious category (1) and +10k rows (34).
        #  if isco % 10 != 0      # Not sure whether to include.
        ]

# ISCO codes with 100 or more rows.
iscover100 = [ isco[0] for isco in iscos if isco[1] > 99 ]

def hilda_by_isco(isco):
    """Get by 2 digit code if `isco` like 21. By 1 digit code if like 2."""
    # Get available iscos - excluding ISCO1 and ISCO34 (dubious or too many).
    iscos = [ int(code) for code in hilda['tjbm682'].unique()
              if int(code) not in [1, 34] ]
    # Inline convenience lookup functions.
    def hilda_by_isco2d(isco_2d): return hilda[hilda['tjbm682'] == isco_2d]
    def first(x): return (x - x % 10) // 10

    # If a 2-digit code is requested.
    if isco > 9:
        return hilda_by_isco2d(isco)
    # If a 1-digit code is requested.
    else:
        return pd.concat([ hilda[hilda['tjbm682'] == code]
                           for code in iscos if first(code) == isco ])




if __name__ == '__main__': pass
