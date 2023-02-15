#!/bin/env python3
#
# hilda.py - for (pre)processing of HILDA data.
#
# Author: Fjalar de Haan (fjalar.dehaan@unimelb.edu.au)
# Created: 2022-09-12
# Last modified: 2022-11-30
#

import pickle

import numpy as np
import pandas as pd
import pyreadstat

# Path strings to 20th ('t') wave of HILDA data set.
hilda_spss_path = "/server/data/hilda/spss-200c/Combined t200c.sav"
hilda_pickle_path = "/server/data/hilda/combined-t200c.pickle"

# with open(hilda_pickle_path, "rb") as f:
    # hilda = pickle.load(f)

hilda, meta = pyreadstat.read_sav(hilda_spss_path)

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


if __name__ == '__main__':
    ... # Do things.



