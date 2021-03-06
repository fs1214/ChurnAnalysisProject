__author__ = 'fangyunsun'

import pandas as pd
import numpy as np
import sys
import re


"""
Read the raw data into dataframe. Drop the empty column.
"""
def read_file(filename):
    df = pd.read_csv(filename,index_col=0)
    df = df.dropna(axis=1,how='all')
    new_df = df.fillna(0)
    return new_df

"""
Remove the digit in industry field
"""
def remove_digit(old_str):
    if old_str.isdigit() == True:
        return 'no industry'
    new_str_li = re.findall(r"\w+", old_str)
    res_li = []
    for item in new_str_li:
        if item.isdigit() == False:
            res_li.append(item.lower())
    return ' '.join(res_li)


"""
Clean the dataframe
"""
def clean_data(df):
    
    #Remove digit in industry field
    df['Industry'] = df['Industry'].apply(lambda x:remove_digit(str(x)))
    industry_dummy = pd.get_dummies(df['Industry'])
    #df = pd.concat([df, industry_dummy], axis=1)

    #Validate RecentContractType/FirstContractType field, e.g. replace 12C with 12M. Create dummy variables from this categorical column and add them to the dataframe
    df.loc[df['RecentContractType']=='12C','RecentContractType']= '12M'
    df.loc[df['RecentContractType']=='14M','RecentContractType']= '12M'
    df.loc[df['RecentContractType']=='36C','RecentContractType']= '36M'
    df.loc[df['RecentContractType']=='M36','RecentContractType']= '36M'
    recent_contract_dummy = pd.get_dummies(df['RecentContractType'])
    recent_contract_dummy.columns = ['No Recent Contract','12M Recent Contract','24M Recent Contract','36M Recent Contract','MTM Recent Contract']
    df = pd.concat([df,recent_contract_dummy], axis=1)

    df.loc[df['FirstContractType']=='12C','FirstContractType']= '12M'
    df.loc[df['FirstContractType']=='14M','FirstContractType']= '12M'
    df.loc[df['FirstContractType']=='M36','FirstContractType']= '36M'
    df.loc[df['FirstContractType']=='36C','FirstContractType']= '36M'
    first_contract_dummy = pd.get_dummies(df['FirstContractType'])
    first_contract_dummy.columns = ['No First Contract','12M First Contract','24M First Contract','36M First Contract','MTM First Contract']
    df = pd.concat([df,first_contract_dummy], axis=1)

    #Substitue the relative value for the absolute value. 
    df['NumOfRepairTickets'] = df['NumOfRepairTickets']/df['AvgMonthlyBilling']
    df['NumOfInboundCalls'] = df['NumOfInboundCalls']/df['AvgMonthlyBilling']
    df['NumOfOutboundCalls'] = df['NumOfOutboundCalls']/df['AvgMonthlyBilling']
    df['InboundCallMinutes'] = df['InboundCallMinutes']/df['AvgMonthlyBilling']
    df['OutboundCallMinutes'] = df['OutboundCallMinutes']/df['AvgMonthlyBilling']
    df['NumOfFCTickets'] = df['NumOfFCTickets']/df['AvgMonthlyBilling']
    df['NumOfHDTickets'] = df['NumOfHDTickets']/df['AvgMonthlyBilling']
    df['NumOfSATickets'] = df['NumOfSATickets']/df['AvgMonthlyBilling']
    df['FCLinesAffected'] = df['FCLinesAffected']/df['AvgMonthlyBilling']
    df['HDLinesAffected'] = df['HDLinesAffected']/df['AvgMonthlyBilling']
    df['SALinesAffected'] = df['SALinesAffected']/df['AvgMonthlyBilling']
    df['FCAvgResolvingDays'] = df['FCAvgResolvingDays']/df['AvgMonthlyBilling']
    df['HDAvgResolvingDays'] = df['HDAvgResolvingDays']/df['AvgMonthlyBilling']
    df['SAAvgResolvingDays'] = df['SAAvgResolvingDays']/df['AvgMonthlyBilling']
    df['CreditAmt'] = df['CreditAmt']/df['AvgMonthlyBilling']
    df['CreditReqCnt'] = df['CreditReqCnt']/df['AvgMonthlyBilling']
    df['RangeOfBilling'] = df['RangeOfBilling']/df['AvgMonthlyBilling']
    
    #Deal with nan and infinity values.
    df.replace(np.inf, 0, inplace=True)
    df.replace(np.nan, 0, inplace=True)

    return df

if __name__ == "__main__":
    
    #In the terminal, enter 'python DataPreprocess.py filename'
    #Clean the data and store the dataset as csv format.
    filename = sys.argv[1]
    df = read_file(filename)
    processed_df = clean_data(df)
    processed_df.to_csv('clean_data_20140701.csv')