#!/bin/env python3
#
# hilda.py - for (pre)processing of HILDA data.
#
# Author: Fjalar de Haan (fjalar.dehaan@unimelb.edu.au)
# Created: 2022-09-12
# Last modified: 2023-03-01
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
        , 'tjbmhrha'
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
        , 'tjbmhruw'
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

# Replace dicts.
age = { 'Less than 1 year': 0.0 }
sex = { 'Female': 0
      , 'Male': 1 }
edu = { 'Doctorate': 14
      , 'Masters Degree': 13
      , 'Graduate Diploma': 12
      , 'Graduate Certificate': 11
      , 'Honours Bachelor Degree': 10
      , 'Bachelor Degree but not Honours': 9
      , 'Advanced Diploma (3 years full-time or equivalent)': 8
      , 'Associate Degree': 7
      , 'Diploma (2 years full-time or equivalent)': 6
      , 'Certificate - Dont know level': 5
      , 'Certificate level IV': 4
      , 'Certificate level III': 3
      , 'Certificate level II': 2
      , 'Certificate level I': 1
      , 'Other': np.nan }
sat = { 'Neither satisfied nor dissatisfied': 5.0
      , 'Totally dissatisfied': 0.0
      , 'Totally satisfied': 10.0 }
dec = { '2nd decile': 20
      , '3rd decile': 30
      , '4th decile': 40
      , '5th decile': 50
      , '6th decile': 60
      , '7th decile': 70
      , '8th decile': 80
      , '9th decile': 90
      , 'Highest decile': 100
      , 'Lowest decile': 10 }

def col_hgage(column):
    return pd.to_numeric(column, errors='coerce').fillna(0)

def col_hgsex(column):
    return column.replace({'Female': 0, 'Male': 1})

def col_tedlhqn(column):
    rank = { 'Doctorate': 14
           , 'Masters Degree': 13
           , 'Graduate Diploma': 12
           , 'Graduate Certificate': 11
           , 'Honours Bachelor Degree': 10
           , 'Bachelor Degree but not Honours': 9
           , 'Advanced Diploma (3 years full-time or equivalent)': 8
           , 'Associate Degree': 7
           , 'Diploma (2 years full-time or equivalent)': 6
           , 'Certificate - Dont know level': 5
           , 'Certificate level IV': 4
           , 'Certificate level III': 3
           , 'Certificate level II': 2
           , 'Certificate level I': 1
           }
    return column.replace(rank).replace('Other', np.NaN)

# Read the HILDA data one way or another.
if os.path.exists(hilda_spss_path):
    raw, meta = pyreadstat.read_sav(hilda_spss_path)
    hilda = clean(raw)
else:
    with open(hilda_pickle_path, "rb") as f:
        hilda, meta = pickle.load(f)

# Produce subsets of HILDA based on Fjalar, Brandon or Josh's columns.
hildaf = hilda[fcols.keys()]
hildab = hilda[bcols]
hildaj = hilda[jcols.keys()]

if __name__ == '__main__': pass
