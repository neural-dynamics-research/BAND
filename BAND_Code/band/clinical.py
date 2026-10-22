import numpy as np
import pandas as pd
from scipy.stats import ttest_rel,linregress

def paired_medication(table, id_col='subject_id',state_col='medication',value_col='beta_power'):
    df=pd.DataFrame(table).copy()
    required={id_col,state_col,value_col}
    if not required.issubset(df.columns):raise ValueError(f'Missing columns: {required-set(df.columns)}')
    df[state_col]=df[state_col].astype(str).str.upper().str.strip()
    df=df.loc[df[state_col].isin(['OFF','ON'])]
    # Multiple sessions within one subject/state are combined before the paired test.
    grouped=df.groupby([id_col,state_col])[value_col].mean().unstack()
    paired=grouped.dropna(subset=['OFF','ON']).copy()
    if len(paired)<2:raise ValueError('Need at least two paired participants')
    change=paired['ON']-paired['OFF']; stat,p=ttest_rel(paired['ON'],paired['OFF'])
    return paired.reset_index(),{'n_paired':int(len(paired)),'mean_ON_minus_OFF':float(change.mean()),
       'sd_difference':float(change.std(ddof=1)),'paired_t':float(stat),'p_two_sided':float(p)}

def clinical_association(table,x_col='mds_updrs_iii',y_col='beta_power',id_col='subject_id'):
    df=pd.DataFrame(table).dropna(subset=[x_col,y_col]);df=df.groupby(id_col)[[x_col,y_col]].mean().reset_index()
    if len(df)<3:raise ValueError('Need >=3 independent participants')
    fit=linregress(df[x_col],df[y_col])
    return df,{'n_participants':len(df),'slope':float(fit.slope),'intercept':float(fit.intercept),
       'pearson_r':float(fit.rvalue),'p_two_sided':float(fit.pvalue),'slope_standard_error':float(fit.stderr)}
