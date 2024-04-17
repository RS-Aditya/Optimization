import pandas as pd
import networkx as nx
import os
import numpy as np
from pathlib import Path
from itertools import product
from ast import literal_eval
import re
import math
from multiprocessing import Pool
from hyperopt import hp, space_eval, tpe, fmin, STATUS_OK

base_loc = r"C:\Users\IN22916394\OneDrive - Tesco\Pricing WorkBench"
# base_loc = r"C:\Users\IN22916618\Tesco\EA_INT_CZ_FORECAST - General\Pricing WorkBench"
input_loc = r"01 Input Data\02 Cleaned Data\1.6 Optimization\CZ"
tpn_list_loc = r'04 Outputs\1.5 BB Optimization\CZ files adi\05 Apr'
intermediate_loc =  r"02 Intermediate Data\1.5 BB Optimization\CZ Opti Base File\05 Apr"
output_loc = r"04 Outputs\1.3 Demand Forecasting\CZ_adi\05 Apr\Coeff"
cat_reset_loc = r"01 Input Data\02 Cleaned Data\1.5 Cat Reset Data\07042024"
promo_loc = r"01 Input Data\02 Cleaned Data\1.4 Back Basket TPN Shortlist\CZ"


base_output_loc = r"04 Outputs\1.6 Opti Run\10 Apr"
tpn_list_loc = r'04 Outputs\1.5 BB Optimization\CZ files adi\05 Apr\01. POC Shortlist'
tpn_aff_list_loc = r'04 Outputs\1.5 BB Optimization\CZ files adi\05 Apr\02. Affinity'
tpn_subs_list_loc = r'04 Outputs\1.5 BB Optimization\CZ files adi\05 Apr\03. Substitution\02. EA'
net_pfam_list_loc = r'04 Outputs\1.5 BB Optimization\CZ files adi\05 Apr'
sales_vol_loc = r'02 Intermediate Data\1.5 BB Optimization\Sales Data\CZ'
promo_loc = r'01 Input Data\02 Cleaned Data\1.4 Back Basket TPN Shortlist\CZ'

# bkt_lines_loc = r'02 Intermediate Data\1.5 BB Optimization\BB Lines'
# cal_df = pd.read_excel(os.path.join(Path(base_loc, input_loc),'Calendar.xlsx'))

# =============================================================================
# tpn price family
# =============================================================================

cat_reset_data = pd.read_excel(os.path.join(Path(base_loc, cat_reset_loc),'Category Reset Output (25).xlsx'), skiprows=1)
tpn_price_family = cat_reset_data[['TPN', 'Price Family', 'OBBrand Tier']]
tpn_price_family.rename(columns = {'Price Family': 'Price_Family'}, inplace = True)
tpn_price_family['Price_Family'] = np.where((tpn_price_family['Price_Family'].isna() | tpn_price_family['Price_Family'] == 0), tpn_price_family['TPN'], tpn_price_family['Price_Family'])
tpn_price_family['Own_Brand'] = np.where(tpn_price_family['OBBrand Tier'].isna()== True, "Brand", "OB")
promo_df = pd.read_excel(os.path.join(Path(base_loc, promo_loc, 'promo_days_0504.xlsx')))
promo_df.rename(columns = {'Row Labels': 'TPN', 'Distinct Count of fiscal_week': 'weeks_on_promo'}, inplace = True)

# =============================================================================
# tesco and compt price
# =============================================================================

tesco_price = pd.read_excel(os.path.join(Path(base_loc, input_loc),'Price Compliance Report w07.xlsx'))
tesco_price = tesco_price[['TPN', 'CURRENT TESCO PRICE', 'CURRENT COST PRICE', 'LIDL', 
                           'KAUFLAND', 'ALBERT', 'GLOBUS', 'PENNY', 'BILLA', 'DMA', 'IKEA',
                           'LID PF', 'KAU PF', 'ALB PF', 'GLO PF', 
                           'PEN PF', 'BIL PF', 'DMA PF', 'IKE PF']]
tesco_price['Lid_final_pr'] = np.where(tesco_price['LIDL'].notna(), tesco_price['LIDL'],
                                        np.where(tesco_price['LID PF'].notna(), tesco_price['LID PF'], tesco_price['LIDL']))
tesco_price['Kau_final_pr'] = np.where(tesco_price['KAUFLAND'].notna(), tesco_price['KAUFLAND'],
                                        np.where(tesco_price['KAU PF'].notna(), tesco_price['KAU PF'], tesco_price['KAUFLAND']))
tesco_price['Alb_final_pr'] = np.where(tesco_price['ALBERT'].notna(), tesco_price['ALBERT'],
                                        np.where(tesco_price['ALB PF'].notna(), tesco_price['ALB PF'], tesco_price['ALBERT']))
tesco_price['Glo_final_pr'] = np.where(tesco_price['GLOBUS'].notna(), tesco_price['GLOBUS'],
                                        np.where(tesco_price['GLO PF'].notna(), tesco_price['GLO PF'], tesco_price['GLOBUS']))
tesco_price['Pen_final_pr'] = np.where(tesco_price['PENNY'].notna(), tesco_price['PENNY'],
                                        np.where(tesco_price['PEN PF'].notna(), tesco_price['PEN PF'], tesco_price['PENNY']))
tesco_price['Bil_final_pr'] = np.where(tesco_price['BILLA'].notna(), tesco_price['BILLA'],
                                        np.where(tesco_price['BIL PF'].notna(), tesco_price['BIL PF'], tesco_price['BILLA']))
tesco_price['DM_final_pr'] = np.where(tesco_price['DMA'].notna(), tesco_price['DMA'],
                                        np.where(tesco_price['DMA PF'].notna(), tesco_price['DMA PF'], tesco_price['DMA']))
tesco_price['IKE_final_pr'] = np.where(tesco_price['IKEA'].notna(), tesco_price['IKEA'],
                                        np.where(tesco_price['IKE PF'].notna(), tesco_price['IKE PF'], tesco_price['IKEA']))

tesco_price.rename(columns = {'CURRENT TESCO PRICE':'Tesco_Regular_Price', 'CURRENT COST PRICE':'Cost_Price'}, inplace = True)


# merge with price family
tpn_price_df = pd.merge(tesco_price, tpn_price_family, how = 'left', left_on='TPN', right_on = 'TPN')

tpn_price_df['Final_Comp_Price'] = np.nan
tpn_price_df['Final_Comp_Price'] = tpn_price_df['Final_Comp_Price'].fillna(tpn_price_df['Kau_final_pr'])
tpn_price_df['Final_Comp_Price'] = tpn_price_df['Final_Comp_Price'].fillna(tpn_price_df['Alb_final_pr'])
tpn_price_df['Final_Comp_Price'] = tpn_price_df['Final_Comp_Price'].fillna(tpn_price_df[['Lid_final_pr', 'Glo_final_pr', 'Bil_final_pr', 'Pen_final_pr', 'DM_final_pr']].max(axis=1))

# Create no price flag
tpn_price_df['No_comp_price_flag'] = np.where(tpn_price_df['Final_Comp_Price'].isna(), 1, 0)

# TPN current price cap current max cap
tpn_price_df['Curr_max_cap'] = tpn_price_df['Tesco_Regular_Price']*1.1
tpn_price_df['Curr_min_cap'] = tpn_price_df['Tesco_Regular_Price']*0.9

# No Cap TPN comp prices
tpn_price_df['Min_Price_no_cap'] = ((tpn_price_df['Final_Comp_Price']/105.0)*100).round(2).fillna(0)
tpn_price_df['Max_Price_no_cap'] = ((tpn_price_df['Final_Comp_Price']/95.0)*100).round(2).fillna(0)

# Capped Min and Max Prices
tpn_price_df['Comp_max_lt_Max_cap'] = np.where(tpn_price_df['Max_Price_no_cap'] < tpn_price_df['Curr_max_cap'], 1, 0)
tpn_price_df['Comp_max_lt_Min_cap'] = np.where(tpn_price_df['Max_Price_no_cap'] < tpn_price_df['Curr_min_cap'], 1, 0)

tpn_price_df['Comp_max_gt_Max_cap'] = np.where(tpn_price_df['Max_Price_no_cap'] > tpn_price_df['Curr_max_cap'], 1, 0)
tpn_price_df['Comp_max_gt_Min_cap'] = np.where(tpn_price_df['Max_Price_no_cap'] > tpn_price_df['Curr_min_cap'], 1, 0)

tpn_price_df['Comp_min_lt_Min_cap'] = np.where(tpn_price_df['Min_Price_no_cap'] < tpn_price_df['Curr_min_cap'], 1, 0)
tpn_price_df['Comp_min_lt_Max_cap'] = np.where(tpn_price_df['Min_Price_no_cap'] < tpn_price_df['Curr_max_cap'], 1, 0)

tpn_price_df['Comp_min_gt_Min_cap'] = np.where(tpn_price_df['Min_Price_no_cap'] > tpn_price_df['Curr_min_cap'], 1, 0)
tpn_price_df['Comp_min_gt_Max_cap'] = np.where(tpn_price_df['Min_Price_no_cap'] > tpn_price_df['Curr_max_cap'], 1, 0)


tpn_price_df['Max_Price_cap'] = np.where(tpn_price_df['No_comp_price_flag'] ==0, np.where((tpn_price_df['Comp_max_lt_Max_cap']), tpn_price_df['Max_Price_no_cap'],
                                         np.where(tpn_price_df['Comp_max_gt_Max_cap'], tpn_price_df['Curr_max_cap'], tpn_price_df['Max_Price_no_cap'])),0)

tpn_price_df['Min_Price_cap'] = np.where(tpn_price_df['No_comp_price_flag'] ==0, np.where((tpn_price_df['Comp_min_lt_Min_cap']), tpn_price_df['Curr_min_cap'],
                                         np.where(tpn_price_df['Comp_min_gt_Min_cap'], tpn_price_df['Min_Price_no_cap'], tpn_price_df['Curr_min_cap'])),0)

# When Comp Min and Comp Max > Max Cap, then Min_Price_Cap should be Current Price and Max_Price_Cap should be Max Cap 
# When Comp Min and Comp Max < Min Cap the Min_Price_Cap should be Min Cap and Max_Price_Cap should be Current Price 

tpn_price_df['Min_Price_cap'] = np.where((tpn_price_df['Comp_min_lt_Min_cap']) & (tpn_price_df['Comp_max_lt_Min_cap']),
                                         tpn_price_df['Curr_min_cap'],tpn_price_df['Min_Price_cap'])

tpn_price_df['Max_Price_cap'] = np.where((tpn_price_df['Comp_min_lt_Min_cap']) & (tpn_price_df['Comp_max_lt_Min_cap']),
                                         tpn_price_df['Tesco_Regular_Price'],tpn_price_df['Max_Price_cap'])
                                          
tpn_price_df['Min_Price_cap'] =  np.where((tpn_price_df['Comp_max_gt_Max_cap']) & (tpn_price_df['Comp_min_gt_Max_cap']),
                                         tpn_price_df['Tesco_Regular_Price'],tpn_price_df['Min_Price_cap'])


tpn_price_df['Max_Price_cap'] = np.where((tpn_price_df['Comp_max_gt_Max_cap']) & (tpn_price_df['Comp_min_gt_Max_cap']),
                                         tpn_price_df['Curr_max_cap'],tpn_price_df['Max_Price_cap'])

                                         
                                          
# tpn_price_df['Min_price_cap_upd'] = np.where(tpn_price_df['Min_Price_cap'] > tpn_price_df['Max_Price_no_cap'], tpn_price_df['Tesco_Regular_Price'],tpn_price_df['Min_Price_cap'])
# tpn_price_df['Max_price_cap_upd'] = np.where(tpn_price_df['Max_Price_cap'] < tpn_price_df['Min_Price_no_cap'], tpn_price_df['Tesco_Regular_Price'],tpn_price_df['Max_Price_cap'])



# tpn_price_df['No_comp_price_flag'] = np.where((tpn_price_df['Comp_max_lt_Min_cap'])& (tpn_price_df['Comp_min_lt_Min_cap']), 1, tpn_price_df['No_comp_price_flag'])
# tpn_price_df['No_comp_price_flag'] = np.where((tpn_price_df['Comp_min_gt_Max_cap'])& (tpn_price_df['Comp_max_gt_Max_cap']), 1, tpn_price_df['No_comp_price_flag'])


# tpn_price_df.to_excel("tpn_price_df.xlsx")


# tpn_price_df['Comp_Index'] = ((tpn_price_df['Final_Comp_Price']/tpn_price_df['Tesco_Regular_Price'])*100).round(2).fillna(0)

# # =IFERROR(CEILING(ROUND(K2/1.05,4), 0.01), "-")
# tpn_price_df['Low_Bound_Index'] = (tpn_price_df['Final_Comp_Price']/1.05).round(4)
# tpn_price_df['Low_Bound_Index'] = np.ceil(tpn_price_df['Low_Bound_Index']/0.01)*0.01
# tpn_price_df['Low_Bound_Index'] = tpn_price_df['Low_Bound_Index'].fillna(0)

# # =IFERROR(FLOOR(ROUND(K2/0.95,4), 0.01), "-")
# tpn_price_df['Up_Bound_Index'] = (tpn_price_df['Final_Comp_Price']/0.95).round(4)
# tpn_price_df['Up_Bound_Index'] = np.floor(tpn_price_df['Up_Bound_Index']/0.01)*0.01
# tpn_price_df['Up_Bound_Index'] = tpn_price_df['Up_Bound_Index'].fillna(0) #small diffenrece in sum with excel

# tpn_price_df['Up_Bound_Index_upd'] = np.where(tpn_price_df['Own_Brand']=='OB', (tpn_price_df['Final_Comp_Price']/1).round(4), tpn_price_df['Up_Bound_Index'] )
# tpn_price_df['Up_Bound_Index_upd'] = np.floor(tpn_price_df['Up_Bound_Index_upd']/0.01)*0.01
# tpn_price_df['Up_Bound_Index_upd'] = tpn_price_df['Up_Bound_Index_upd'].fillna(0) #small diffenrece in sum with excel

# tpn_price_df.drop(columns = ['Up_Bound_Index'], inplace = True)
# tpn_price_df.rename(columns = {'Up_Bound_Index_upd':'Up_Bound_Index'}, inplace = True)

# =============================================================================
# TPN Role
# =============================================================================

promo_df = pd.read_excel(os.path.join(Path(base_loc, promo_loc, 'promo_days_0504.xlsx')))
promo_df.rename(columns = {'Row Labels': 'TPN', 'Distinct Count of fiscal_week': 'weeks_on_promo'}, inplace = True)
tpn_list = pd.read_excel(os.path.join(Path(base_loc, net_pfam_list_loc, 'NetTPNList_withPFam.xlsx')))
tpn_list.rename(columns = {'NetTPNwithPFam': 'TPN'}, inplace = True)
price_change_na_df = pd.read_excel(os.path.join(Path(base_loc, input_loc),'List of No Change and No premium.xlsx'))
price_family_poc = tpn_price_family[tpn_price_family['TPN'].isin(tpn_list['TPN'])]
price_family_poc = price_family_poc[price_family_poc['Price_Family'].notna()][['Price_Family']]
tpn_list_w_pfam = tpn_price_family[tpn_price_family['Price_Family'].isin(price_family_poc['Price_Family'])][['TPN']]
tpn_list_new = tpn_list.append(tpn_list_w_pfam)
tpn_list_new.drop_duplicates(inplace = True)
tpn_list_new = tpn_list_new[~(tpn_list_new['TPN'].isin(promo_df['TPN']))]
tpn_list_new = tpn_list_new[~(tpn_list_new['TPN'].isin(price_change_na_df['TPN']))]

# Keep only TPNs present in Test stores
str_pres_df = pd.read_excel(os.path.join(Path(base_loc, input_loc),'store_presence_long.xlsx'))
str_pres_df = str_pres_df[str_pres_df['Store Code'].isin([11005,11008,11010,11017,11028,11043,11015,11049,11057,11145,11063,11046,11009,11047,11040,11053])]
str_pres_df = str_pres_df[str_pres_df['Status'] == 'C']
tpn_list_new = tpn_list_new[tpn_list_new['TPN'].isin(str_pres_df['TPN'])]

tpn_list_new = tpn_list_new.merge(cat_reset_data[['TPN', 'CIS', 'Elasticity']], on = 'TPN', how = 'left')
tpn_list_new['CIS_percentile'] = tpn_list_new['CIS'].rank(pct=True)
tpn_list_new['Elasticity_percentile'] = tpn_list_new['Elasticity'].rank(pct=True)
cis_percentile_50 = tpn_list_new['CIS'].quantile(0.5)
elas_percentile_50 = tpn_list_new['Elasticity'].quantile(0.5)
tpn_list_new['CIS_50%tile'] = cis_percentile_50
tpn_list_new['Elas_50%tile'] = elas_percentile_50

#Categorise into low, high Cis and low high ealsticity flags
tpn_list_new['LOW_CIS'] = np.where(tpn_list_new['CIS'] < cis_percentile_50, 1, 0)
tpn_list_new['HIGH_CIS'] = np.where(tpn_list_new['CIS'] >= cis_percentile_50, 1, 0)

tpn_list_new['LOW_Elasticity'] = np.where(tpn_list_new['Elasticity'] < elas_percentile_50, 1, 0)
tpn_list_new['HIGH_Elasticity'] = np.where(tpn_list_new['Elasticity'] >= elas_percentile_50, 1, 0)

tpn_list_new['Role'] = np.where((tpn_list_new['HIGH_Elasticity'] == 1) & (tpn_list_new['HIGH_CIS'] == 1), "Attack", 
                                np.where((tpn_list_new['HIGH_CIS'] == 1) & (tpn_list_new['LOW_Elasticity'] == 1), "Defend",
                                         np.where((tpn_list_new['HIGH_Elasticity'] == 1) & (tpn_list_new['LOW_CIS'] == 1), "Defend", 
                                                  np.where((tpn_list_new['LOW_Elasticity'] == 1) & (tpn_list_new['LOW_CIS'] == 1), "Profit", "Attack"))))




tpn_price_df = pd.merge(tpn_price_df, tpn_list_new, how = 'left', left_on = 'TPN', right_on = 'TPN')

tpn_price_df['Low_Bound_Role'] = np.where(tpn_price_df['No_comp_price_flag'] == 1,
                                          np.where(tpn_price_df['Role'] == "Attack", tpn_price_df['Tesco_Regular_Price']-(tpn_price_df['Tesco_Regular_Price']*0.05),
                                                   np.where(tpn_price_df['Role'] == "Defend", tpn_price_df['Tesco_Regular_Price']-(tpn_price_df['Tesco_Regular_Price']*0.03),
                                                            np.where(tpn_price_df['Role'] == "Profit", tpn_price_df['Tesco_Regular_Price'], 0))),0)


tpn_price_df['Low_Bound_Role'] = tpn_price_df['Low_Bound_Role'].round(4)
tpn_price_df['Low_Bound_Role'] = np.ceil(tpn_price_df['Low_Bound_Role']/0.01) * 0.01
tpn_price_df['Low_Bound_Role'] = tpn_price_df['Low_Bound_Role'].fillna(0) #3.0 difference with excel


tpn_price_df['Up_Bound_Role'] = np.where(tpn_price_df['No_comp_price_flag'] == 1,
                                          np.where(tpn_price_df['Role'] == "Attack", tpn_price_df['Tesco_Regular_Price']+(tpn_price_df['Tesco_Regular_Price']*0.05),
                                                   np.where(tpn_price_df['Role'] == "Defend", tpn_price_df['Tesco_Regular_Price']+(tpn_price_df['Tesco_Regular_Price']*0.03),
                                                            np.where(tpn_price_df['Role'] == "Profit", tpn_price_df['Tesco_Regular_Price']+(tpn_price_df['Tesco_Regular_Price']*0.05), 0))),0)

tpn_price_df['Up_Bound_Role'] = tpn_price_df['Up_Bound_Role'].round(4)
tpn_price_df['Up_Bound_Role'] = np.floor(tpn_price_df['Up_Bound_Role']/0.01) * 0.01
tpn_price_df['Up_Bound_Role'] = tpn_price_df['Up_Bound_Role'].fillna(0)

tpn_price_df['Low_Bound_Final'] = np.where(tpn_price_df['No_comp_price_flag'] == 1, tpn_price_df['Low_Bound_Role'], tpn_price_df['Min_Price_cap'])
tpn_price_df['Up_Bound_Final'] = np.where(tpn_price_df['No_comp_price_flag'] == 1, tpn_price_df['Up_Bound_Role'], tpn_price_df['Max_Price_cap'])

tpn_price_df['Price_Bound_Key'] = tpn_price_df.apply(lambda row: (row['Low_Bound_Final'], row['Up_Bound_Final']), axis = 1)


# =============================================================================
# Function for Rounding rules
# =============================================================================
def rounding_rule(low_bound, up_bound, replace_value):
    
    
    # low_bound = 25
    # up_bound = 30
    final_val = []
    
    # if (low_bound < 0 or up_bound > 10^4):
    #     return 0
    
    if (low_bound < 5):
        if up_bound >= 5:
            val_accept = list(np.arange(0.1, 5, 0.1).round(2))
            val_accept_updated = [v for v in val_accept if (v>=round(low_bound,2))]
            final_val.extend(val_accept_updated)
        else:
            val_accept = list(np.arange(0.1, 5, 0.1).round(2))
            val_accept_updated = [v for v in val_accept if (v>=round(low_bound,2)) and (v<=round(up_bound,2))]
            final_val.extend(val_accept_updated)
            
    if (low_bound >= 5 and low_bound < 50) or (up_bound >= 5 ):
        if up_bound>=50:
            val_accept_5 = np.arange(5,50,0.5).round(2)
            val_accept_9 = np.arange(5,50,0.1).round(2)
            val_accept_5_updated = [v for v in val_accept_5 if ('.5' in str(v)) and (v>=round(low_bound,2))]
            val_accept_9_updated = [v for v in val_accept_9 if ('.9' in str(v)) and (v>=round(low_bound,2))]
            final_val.extend(val_accept_5_updated)
            final_val.extend(val_accept_9_updated)
        else:
            val_accept_5 = np.arange(5,50,0.5).round(2)
            val_accept_9 = np.arange(5,50,0.1).round(2)
            val_accept_5_updated = [v for v in val_accept_5 if ('.5' in str(v)) and (v>=round(low_bound,2)) and (v<=round(up_bound,2))]
            val_accept_9_updated = [v for v in val_accept_9 if ('.9' in str(v)) and (v>=round(low_bound,2)) and (v<=round(up_bound,2))]
            final_val.extend(val_accept_5_updated)
            final_val.extend(val_accept_9_updated)
            
    if (low_bound >= 50 and low_bound < 100) or (up_bound >= 50):
        if up_bound >= 100:
            val_accept_09 = np.arange(50,100,0.1).round(2)
            val_accept_09_updated = [v for v in val_accept_09 if ('0.9' not in str(v) and '.9' in str(v)) and (v>=round(low_bound,2))]
            final_val.extend(val_accept_09_updated)
        else:    
            val_accept_09 = np.arange(50,100,0.1).round(2)
            val_accept_09_updated = [v for v in val_accept_09 if ('0.9' not in str(v) and '.9' in str(v)) and (v>=round(low_bound,2)) and (v<=round(up_bound,2))]
            final_val.extend(val_accept_09_updated)
                        
    if (low_bound >= 100 and low_bound < 300) or (up_bound >= 100):
        if up_bound >= 300:
            val_accept_49 = list(np.arange(100,300,0.1).round(2))
            val_accept_99 = list(np.arange(100,300,0.1).round(2))
            val_accept_49_updated = [v for v in val_accept_49 if ('4.9' in str(v)) and (v>=round(low_bound,2))]
            val_accept_99_updated = [v for v in val_accept_99 if ('9.9' in str(v)) and (v>=round(low_bound,2))]
            final_val.extend(val_accept_49_updated)
            final_val.extend(val_accept_99_updated)
        else:
           val_accept_49 = list(np.arange(100,300,0.1).round(2))
           val_accept_99 = list(np.arange(100,300,0.1).round(2))
           val_accept_49_updated = [v for v in val_accept_49 if ('4.9' in str(v)) and (v>=round(low_bound,2)) and (v<=round(up_bound,2))]
           val_accept_99_updated = [v for v in val_accept_99 if ('9.9' in str(v)) and (v>=round(low_bound,2)) and (v<=round(up_bound,2))]
           final_val.extend(val_accept_49_updated)
           final_val.extend(val_accept_99_updated)
    
    if (low_bound >= 300 and low_bound < 500) or (up_bound >= 300 ):
        if up_bound>=500:
            val_accept_909 = np.arange(300,500,0.1).round(2)
            val_accept_909_updated = [v for v in val_accept_909 if ('9.9' in str(v)) and (v>=round(low_bound,2))]
            final_val.extend(val_accept_909_updated)
            
        else:
            val_accept_909 = np.arange(300,500,0.1).round(2)
            val_accept_909_updated = [v for v in val_accept_909 if ('9.9' in str(v)) and (v>=round(low_bound,2)) and (v<=round(up_bound,2))]
            final_val.extend(val_accept_909_updated)
    
    if (low_bound >= 500 and low_bound < 1000) or (up_bound >= 500):
        if up_bound >= 1000:
            val_accept_90 = np.arange(500,1000,1.00).round(2)
            val_accept_90_updated = [v for v in val_accept_90 if ('09' not in str(v) and '9.0' in str(v)) and (v>=round(low_bound,2))]
            final_val.extend(val_accept_90_updated)
        else:    
            val_accept_90 = np.arange(500,1000,1.00).round(2)
            val_accept_90_updated = [v for v in val_accept_90 if ('09' not in str(v) and '9.0' in str(v)) and (v>=round(low_bound,2)) and (v<=round(up_bound,2))]
            final_val.extend(val_accept_90_updated)
    
        
    if (up_bound >= 1000):
        val_accept_990_up = list(np.arange(1000,up_bound,1.00).round(2))
        val_accept_990_uptd = [v for v in val_accept_990_up if ('99.0' in str(v)) and (v>=round(low_bound,2)) and (v<=round(up_bound,2))]
        final_val.extend(val_accept_990_uptd)
        
    # Replace with specified value if final_val is empty
    if not final_val:
       final_val.append(replace_value)
   
    final_val = sorted(final_val) 
   
    return final_val    

# test = rounding_rule(50, 55, 50)


tpn_price_df['Price_Ranges'] = tpn_price_df.apply(lambda row: rounding_rule(row['Low_Bound_Final'], row['Up_Bound_Final'], row['Tesco_Regular_Price']), axis = 1)

# tpn_price_df.to_excel("tpn_price_bounds.xlsx")


# =============================================================================
# Get the future estimated demand using LFL sales and current year last two months growth
# =============================================================================
demand_future_df = pd.read_csv(os.path.join(Path(base_loc, input_loc),'Demand w09_w13_v1.csv'))
demand_future_df = demand_future_df.rename(columns = {"DEMAND_FORECAST":"future_demand"})


# =============================================================================
# Get the coeffiecients run from Lin Reg Model
# =============================================================================
# read the base optimisation file
optimization_base = pd.read_excel(os.path.join(Path(base_loc, intermediate_loc),'Captain_AssocTPNs_BB_Opti.xlsx'))
optimization_base = optimization_base[~(optimization_base['TPN'].isin(price_change_na_df['TPN']))]
optimization_base = optimization_base[~(optimization_base['Assoc_TPN'].isin(price_change_na_df['TPN']))]
optimization_base = optimization_base[~(optimization_base['AssocCap_TPN'].isin(price_change_na_df['TPN']))]

optimization_df = optimization_base[['TPN', 'AssocCap_TPN', 'Coeff', 'ModCoeff_Flag', 'CoeffMod', 'Prim_TPN', 'Subs_Flag', 'AffPass_Flag', 'AffDriv_Flag', 'PriceFam']]
optimization_df.rename(columns = {'AssocCap_TPN': 'Assoc_TPN'}, inplace = True)
optimization_df['Match'] = np.where(optimization_df['TPN'] == optimization_df['Assoc_TPN'], "1", "0")

# get the associates file
associates_df = optimization_df[optimization_df['Match'] == "0"][['Assoc_TPN']]
associates_df1 = associates_df[~associates_df['Assoc_TPN'].isin(optimization_df['TPN'])].drop_duplicates()

associates_df1['TPN'] = associates_df1['Assoc_TPN']
associates_df1['Coeff'] = np.nan
associates_df1['ModCoeff_Flag'] = np.nan
associates_df1['Basket'] = np.nan
associates_df1['Prim_TPN'] = np.nan
associates_df1['Subs_Flag'] = np.nan
associates_df1['AffPass_Flag'] = np.nan
associates_df1['AffDriv_Flag'] = np.nan
associates_df1 = pd.merge(associates_df1, tpn_price_family[['TPN', 'Price_Family']], on = 'TPN', how = 'left')
associates_df1.rename(columns = {'Price_Family': 'PriceFam'}, inplace = True)
associates_df1['PriceFam'] = np.where(associates_df1['PriceFam'].isna() == True, associates_df1['TPN'], associates_df1['PriceFam'])
associates_df1['Match'] = 0.5  ## Mapping for aasociated TPNs added to the bb_opti_base_file

abc = list(associates_df1['TPN'].unique())

# Iterate through all files in the specified directory
result_df = [] #pd.DataFrame()
for tpn in abc:
    try:
        filename = '{}_LinCoeff.xlsx'.format(tpn)    
        file_path = os.path.join(Path(base_loc, output_loc, filename))
        if os.path.isfile(file_path):
            # Load the file into a DataFrame
            df = pd.read_excel(file_path)
        
            df1 = df[df['Unnamed: 0'] == "current_ref_open_price"]
            df1 = df1[['TPN', 'coef']]
            result_df.append(df1)
    except:
        continue

result_df = pd.concat(result_df)
result_df['CoeffMod'] = np.where(result_df['coef'] > -0.5, -0.5, np.where(result_df['coef'] < -5, -5, result_df['coef']))
result_df['CoeffMod'] = np.where(result_df['coef'] > -0.5, -0.5, np.where(result_df['coef'] < -5, -5, result_df['coef']))
    
coeff_df = result_df.copy()   
coeff_df = result_df.merge(tpn_price_family, on = 'TPN', how = 'left' )
associates_df1 = pd.merge(associates_df1, coeff_df[['TPN', 'CoeffMod']], on = 'TPN', how = 'left')


associates_df2 = associates_df1[['TPN', 'Assoc_TPN', 'Coeff', 'ModCoeff_Flag', 'CoeffMod', 'Basket', 'Prim_TPN', 'Subs_Flag', 
                                 'AffPass_Flag', 'AffDriv_Flag', 'PriceFam', 'Match']]

optimization_df1 = optimization_df.append(associates_df2).reset_index()
optimization_df1.drop(columns = 'index', inplace = True)
optimization_df1['CoeffMod'] = np.where(optimization_df1['CoeffMod'].isna()== True, -0.5, optimization_df1['CoeffMod'])
opti_df = pd.merge(optimization_df1, tpn_price_df[['TPN', 'Tesco_Regular_Price', 'Cost_Price']], how = 'left', on = 'TPN' )
opti_df1 = pd.merge(opti_df, tpn_price_df[['TPN', 'Tesco_Regular_Price', 'Low_Bound_Final', 'Up_Bound_Final']], how = 'left', left_on = 'Assoc_TPN', right_on = 'TPN')
opti_df1_3 = pd.merge(opti_df1, tpn_list_new, how = 'left', left_on = 'Assoc_TPN', right_on = 'TPN' )
opti_df1_3.drop(columns = 'TPN', inplace = True)
opti_df2 = pd.merge(opti_df1_3, demand_future_df[['TPN', 'future_demand']], how = 'left', left_on = 'TPN_x', right_on = 'TPN')


 
opti_df3 = opti_df2[['TPN_x', 'Assoc_TPN','CoeffMod','Match','Tesco_Regular_Price_y','Tesco_Regular_Price_x','Cost_Price',
 'future_demand','Low_Bound_Final','Up_Bound_Final','Basket','Prim_TPN', 'Subs_Flag','AffPass_Flag','AffDriv_Flag', 'PriceFam']]


# Extract unique values from both columns
unique_values = pd.unique(opti_df3[['TPN_x', 'Assoc_TPN']].values.ravel('K'))

# Create a DataFrame with an index for unique values
index_df = pd.DataFrame({'Index': range(1, len(unique_values)+1)}, index=unique_values)
index_df.reset_index(inplace = True)
index_df.rename(columns = {'index': 'TPN'}, inplace = True)


# get the index for the Prim TPN and Assoc TPN
opti_df4 = pd.merge(opti_df3, index_df, how = 'left', left_on = 'TPN_x', right_on = 'TPN' )
opti_df5 = pd.merge(opti_df4, index_df, how = 'left', left_on = 'Assoc_TPN', right_on = 'TPN' )


opti_df6 = opti_df5[['TPN_x', 'Assoc_TPN','CoeffMod','Match','Tesco_Regular_Price_y','Tesco_Regular_Price_x','Cost_Price',
 'future_demand','Low_Bound_Final','Up_Bound_Final', 'Index_y', 'Index_x', 'Basket','Prim_TPN', 'Subs_Flag','AffPass_Flag','AffDriv_Flag', 'PriceFam']]


opti_df6.columns = ['TPN', 'Drop_TPN', 'Assoc_TPN', 'Coeff', 'Match', 'Current_Price', 'Current_Price_Main', 'Cost_Main', 'Demand_Main', 'Lower_Bound',
                   'Upper_Bound', 'Index', 'Index_Main', 'Basket','Prim_TPN', 'Subs_Flag','AffPass_Flag','AffDriv_Flag', 'PriceFam']

opti_df7 = opti_df6[['TPN', 'Assoc_TPN', 'Coeff', 'Match', 'Current_Price', 'Current_Price_Main', 'Cost_Main', 'Demand_Main', 'Lower_Bound',
                   'Upper_Bound', 'Index', 'Index_Main', 'Basket','Prim_TPN', 'Subs_Flag','AffPass_Flag','AffDriv_Flag', 'PriceFam']]


# Flag invalid rows
opti_df7['Validity'] = np.where(opti_df7['Current_Price'].isna() == True, "Invalid - No Price", 
                                np.where(opti_df7['Cost_Main'].isna() == True, "Invalid - No Cost", 
                                         np.where(opti_df7['Demand_Main'].isna() == True, "Invalid - No Demand", 
                                                  np.where(opti_df7['Lower_Bound'].isna() == True, "Invalid - No Lower Bound", "Valid"))))
opti_df8 = opti_df7[opti_df7['Validity'] == 'Valid']
opti_df8.drop_duplicates(inplace = True)
data = opti_df8[~((opti_df8['Match'] == 1.5) | (opti_df8['Match'] == 2.5))] #[['TPN', 'Assoc_TPN']] # exclude associate PF and Prim PF
data.rename(columns = {'Assoc_TPN': 'Associated_TPN'}, inplace = True)



# Create a DataFrame
df = data.copy()

# =============================================================================
# Creating exclusive clusters considering alllinked TPNs in one cluster
# =============================================================================

df['cluster'] = df.groupby('TPN').ngroup()
df['cluster_1'] = df.groupby('Associated_TPN')['cluster'].transform('first')
df['cluster_final'] = df.groupby('cluster')['cluster_1'].transform('min')

# Update 'cluster_final' based on the minimum 'cluster_1' for each 'assoc_tpn'
from tqdm import tqdm
for i in tqdm(range(1000)):
    df['cluster_final'] = df.groupby('Associated_TPN')['cluster_final'].transform('min')
    df['cluster_final'] = df.groupby('TPN')['cluster_final'].transform('min')

cluster_df = df.copy()
cluster_df.drop(columns = ['cluster', 'cluster_1'], inplace = True)
cluster_df.rename(columns = {'cluster_final': 'Cluster'}, inplace = True)
cluster_df['Cluster'] = "Cluster_" + cluster_df['Cluster'].astype(str)
cluster_df.drop_duplicates(inplace = True)

cluster_df_upd = cluster_df.rename(columns = {'Associated_TPN': 'Assoc_TPN'})


# Add price ranges to operate on

cluster_df_upd['Price_Ranges'] = cluster_df_upd.apply(lambda row: rounding_rule(row['Lower_Bound'], row['Upper_Bound'], row['Current_Price']), axis = 1)
opti_tobe_tpn_list = cluster_df_upd['Assoc_TPN'].unique().tolist()




######################################################################################################
#### Get all coeffs in 1 Dataframe
######################################################################################################

### Read Subs and Affinity Data
bb_aff_pass_tpn = pd.read_excel(os.path.join(Path(base_loc, tpn_aff_list_loc, "EA_Aff_Result.xlsx")))
bb_aff_driv_tpn = pd.read_excel(os.path.join(Path(base_loc, tpn_aff_list_loc, "EA_AffDriver_Result.xlsx")))
bb_subs_tpn = pd.read_excel(os.path.join(Path(base_loc, tpn_subs_list_loc, "EA_Subs_Result.xlsx")))

## Get the TPN list on which model runs are needed
tpn_run = pd.read_excel(os.path.join(Path(base_loc, net_pfam_list_loc, 'NetTPNList_withPFam.xlsx')))
tpn_run = list(tpn_run['NetTPNwithPFam'])
# tpn_run = tpn_run + [2002120033988, 2002120034025]


feat2assoc_map_df_list = []
assoc_df_list = []

#################
####Get Associate TPNs
#################
for bb_poc_iter in tpn_run: 

    try:
        
        #################
        ### Acquire Subs TPNs Details
        #################
        iter_subs = bb_subs_tpn[bb_subs_tpn['Prim TPN'] == bb_poc_iter]
        iter_subs_list = iter_subs['EA Subs TPN'].unique()
        
        #################
        ### Acquire Passenger Affinity TPNs Details
        #################
        iter_aff_pass = bb_aff_pass_tpn[bb_aff_pass_tpn['Prim TPN'] == bb_poc_iter]
        iter_aff_pass_list = iter_aff_pass['Comp TPN'].unique()
        
        #################
        ### Acquire Driver Affinity TPNs Details
        #################
        iter_aff_driv = bb_aff_driv_tpn[bb_aff_driv_tpn['Prim TPN'] == bb_poc_iter]
        
        # selecting top 3 aff_driver
        iter_aff_driv['lift_rank_new'] = iter_aff_driv.groupby(by = ['Prim TPN'])['Lift'].rank(ascending = False)
        iter_aff_driv = iter_aff_driv[iter_aff_driv['lift_rank_new'] <=3]
        iter_aff_driv_list = iter_aff_driv['Driver TPN'].unique()
        
        ## Create a df for the Asscoiate TPNs - Subs
        subs_iter_df = pd.DataFrame({'assoc_tpn':iter_subs_list})
        subs_iter_df['feature'] = subs_iter_df.index
        subs_iter_df['feature'] = "subs"+ (subs_iter_df['feature']+1).astype(str) +"_price"
        
        ## Create a df for the Asscoiate TPNs - Affinity Passenger
        aff_pass_iter_df = pd.DataFrame({'assoc_tpn':iter_aff_pass_list})
        aff_pass_iter_df ['feature'] = aff_pass_iter_df .index
        aff_pass_iter_df ['feature'] = "AffPass"+ (aff_pass_iter_df ['feature']+1).astype(str) +"_price"
        
        ## Create a df for the Asscoiate TPNs - Affinity Driver
        aff_driv_iter_df = pd.DataFrame({'assoc_tpn':iter_aff_driv_list})
        aff_driv_iter_df ['feature'] = aff_driv_iter_df .index
        aff_driv_iter_df ['feature'] = "AffDriv"+ (aff_driv_iter_df ['feature']+1).astype(str) +"_price"
        
        #collate all in one dataframe
        assoc_iter_df = pd.concat([subs_iter_df, aff_pass_iter_df, aff_driv_iter_df])
        assoc_iter_df['assoc_tpn'] = assoc_iter_df['assoc_tpn'].astype(str)
        
        #### Read the coeff data
        coeff_iter_df = pd.read_excel(os.path.join(Path(base_loc, output_loc, str(bb_poc_iter) + "_LinCoeff.xlsx")))
        coeff_iter_df.rename(columns={coeff_iter_df.columns[0]: "feature" }, inplace = True)
        coeff_iter_df['TPN'] = coeff_iter_df['TPN'].astype(str)

        #Merge Featire with there corresponding TPNs    
        coeff_iter_df = coeff_iter_df.merge(assoc_iter_df, on = ['feature'], how ="left" )
        coeff_iter_df['assoc_tpn'] = np.where(coeff_iter_df['feature'] == "current_ref_open_price", coeff_iter_df['TPN'], coeff_iter_df['assoc_tpn'])
        
        #Get Relevant Columns
        feat_assoc_iter_df = coeff_iter_df[['feature','coef','TPN',"assoc_tpn"]]
        feat2assoc_map_df_list.append(feat_assoc_iter_df)
        
        ### Assoc TPN Masters
        assoc_iter_df['Primary TPN'] = bb_poc_iter
        assoc_df_list.append(assoc_iter_df) 
                
    except:
        
        print("No Results for TPN ", bb_poc_iter,"'s as Shelf Price is Constant in the History" )


feat2assoc_map_df = pd.concat(feat2assoc_map_df_list)
feat2assoc_map_df = feat2assoc_map_df[feat2assoc_map_df['assoc_tpn'].notna()]
feat2assoc_map_df.reset_index(drop=True, inplace=True)

#### Create Flags
feat2assoc_map_df['Prim_TPN'] = np.where(feat2assoc_map_df['feature'].str.contains("current_ref_open_price"), 1, 0)
feat2assoc_map_df['Subs_Flag'] = np.where(feat2assoc_map_df['feature'].str.contains("subs"), 1, 0)
feat2assoc_map_df['AffPass_Flag'] = np.where(feat2assoc_map_df['feature'].str.contains("AffPass"), 1, 0)
feat2assoc_map_df['AffDriv_Flag'] = np.where(feat2assoc_map_df['feature'].str.contains("AffDriv"), 1, 0)

### Prim TPNs
feat2assoc_map_df['coefMod'] = np.where((feat2assoc_map_df['Prim_TPN'] == 1) & (feat2assoc_map_df['coef'] < -5), -5, feat2assoc_map_df['coef'])
feat2assoc_map_df['coefMod'] = np.where((feat2assoc_map_df['Prim_TPN'] == 1) & (feat2assoc_map_df['coef'] > -0.5), -0.5, feat2assoc_map_df['coefMod'])

### Substitute TPNs
feat2assoc_map_df['coefMod'] = np.where((feat2assoc_map_df['Subs_Flag'] == 1) & (feat2assoc_map_df['coef'] > 5), 5, feat2assoc_map_df['coefMod'])
feat2assoc_map_df['coefMod'] = np.where((feat2assoc_map_df['Subs_Flag'] == 1) & (feat2assoc_map_df['coef'] < 0.5), 0.5, feat2assoc_map_df['coefMod'])

### Aff Pass TPNs
feat2assoc_map_df['coefMod'] = np.where((feat2assoc_map_df['AffPass_Flag'] == 1) & (feat2assoc_map_df['coef'] < -5), -5, feat2assoc_map_df['coefMod'])
feat2assoc_map_df['coefMod'] = np.where((feat2assoc_map_df['AffPass_Flag'] == 1) & (feat2assoc_map_df['coef'] > -0.3), -0.3, feat2assoc_map_df['coefMod'])

### Aff Pass TPNs
feat2assoc_map_df['coefMod'] = np.where((feat2assoc_map_df['AffDriv_Flag'] == 1) & (feat2assoc_map_df['coef'] < -5), -5, feat2assoc_map_df['coefMod'])
feat2assoc_map_df['coefMod'] = np.where((feat2assoc_map_df['AffDriv_Flag'] == 1) & (feat2assoc_map_df['coef'] > -0.3), -0.3, feat2assoc_map_df['coefMod'])

#Flag with Mod Coeff
feat2assoc_map_df['ModCoeff_Flag'] = np.where(feat2assoc_map_df['coefMod'] != feat2assoc_map_df['coef'], 1, 0)


## Get Basket Mappin for the TPNs
# bskt_df = pd.read_excel(os.path.join(Path(base_loc,bkt_lines_loc,"SK - BB lines.xlsx")), sheet_name = "Sheet1")
bskt_map_df = cat_reset_data[['TPN','Basket Layer']].drop_duplicates()
bskt_map_df.columns = ['assoc_tpn','basket']
bskt_map_df['assoc_tpn'] = bskt_map_df['assoc_tpn'].astype(str)
bskt_map_df['basket'] = np.where(((bskt_map_df['basket'] == "LPG") | (bskt_map_df['basket'] == "BB")),
                                 bskt_map_df['basket'], "FB")

### FB and LPG TPN List
fb_lpg_tpn_list = bskt_map_df[(bskt_map_df['basket'] == "LPG") | (bskt_map_df['basket'] == "FB")]['assoc_tpn'].unique().tolist()
bb_tpn_list = bskt_map_df[(bskt_map_df['basket'] == "BB")]['assoc_tpn'].unique().tolist()
feat2assoc_map_df['BBFlag'] = np.where(feat2assoc_map_df['assoc_tpn'].isin(bb_tpn_list), 1, 0)

## Replace Assoc TPNs with Captain TPNs
tpn_pfam_map_df = cat_reset_data[['TPN','Price Family']]
tpn_pfam_map_df.columns = ['assoc_tpn', 'PriceFam']
tpn_pfam_map_df['assoc_tpn'] = tpn_pfam_map_df['assoc_tpn'].astype(str)
feat2assoc_map_df = feat2assoc_map_df.merge(tpn_pfam_map_df, on = "assoc_tpn", how = "left")
feat2assoc_map_df['PriceFam'] = np.where(feat2assoc_map_df['PriceFam'].isnull(), feat2assoc_map_df['assoc_tpn'], feat2assoc_map_df['PriceFam'])

##Get PFam Capain Mapping
cap_pfam_loc = r'04 Outputs\1.4 Back Basket TPN Shortlist\CZ'
cap_pfam_df = pd.read_excel(os.path.join(Path(base_loc,cap_pfam_loc,"base_captain_flagged_cz.xlsx")))

tpn_cap_pfam_map_df = cap_pfam_df[['slad_tpn', 'Price Family', 'Captain_Flag']].drop_duplicates()
tpn_cap_pfam_map_df = tpn_cap_pfam_map_df[tpn_cap_pfam_map_df['Captain_Flag'] == 1][['slad_tpn', 'Price Family']].drop_duplicates() 
tpn_cap_pfam_map_df.columns = ['assocCap_tpn', 'PriceFam']
tpn_cap_pfam_map_df['assocCap_tpn'] = tpn_cap_pfam_map_df['assocCap_tpn'].astype(str)
feat2assoc_map_df = feat2assoc_map_df.merge(tpn_cap_pfam_map_df, on = "PriceFam", how = "left")
feat2assoc_map_df['assocCap_tpn'] = np.where(feat2assoc_map_df['assocCap_tpn'].isnull(), feat2assoc_map_df['assoc_tpn'], feat2assoc_map_df['assocCap_tpn'])

feat2assoc_map_df = feat2assoc_map_df[feat2assoc_map_df['BBFlag'] == 1]
# feat2assoc_map_df = feat2assoc_map_df[feat2assoc_map_df['TPN'].isin(opti_tobe_tpn_list)]

abc = feat2assoc_map_df[['TPN', 'assocCap_tpn']]
abc_dups = abc.duplicated()
abc_dups = abc_dups[abc_dups == 1].index.tolist()

# abc_dups = abc.iloc[abc_dups, :]
miss_tpn = [tpn for tpn in opti_tobe_tpn_list if tpn not in pd.to_numeric(feat2assoc_map_df['TPN']).unique().tolist()]

######################################################################################################
### Optimization equation formation - Base data
######################################################################################################
import time
start_time = time.time() ### Record Start Time

pricing_opt_net_result = []
clust_list = cluster_df_upd['Cluster'].unique().tolist()#[35:]

# clust_list = [clust_iter for clust_iter in clust_list if clust_iter != "Cluster_0"]
# clust_list = [clust_iter for clust_iter in clust_list if clust_iter != "Cluster_43"]

# clust_list = ['Cluster_14'] #, "Cluster_337"]

for clust_iter in clust_list:
    
    print("#####################")
    print("""""", clust_iter)    
    print("#####################")
    clust_tpn_list = list(cluster_df_upd[cluster_df_upd['Cluster'] == clust_iter]['Assoc_TPN'].unique())
    clust_tpn_list = [tpn for tpn in clust_tpn_list if tpn not in fb_lpg_tpn_list]
    clust_tpn_df = pd.DataFrame(clust_tpn_list)
    clust_tpn_df.columns = ['assocCap_tpn']
    clust_tpn_df['assocCap_tpn'] = clust_tpn_df['assocCap_tpn'].astype(str)
    
    try:
    
        #################
        #### Get the fixed matrices for simulation
        #################
        fut_demand_list = []
        curr_price_list = []
        curr_cost_list = []
        
        clust_props_df = demand_future_df[demand_future_df['TPN'].isin(clust_tpn_list)][['TPN','future_demand']]
        clust_props_df = clust_props_df.merge(tpn_price_df[['TPN', 'Tesco_Regular_Price', 'Cost_Price']], right_on = "TPN", left_on = "TPN", how = "left")
        clust_props_df = clust_props_df.sort_values(['TPN'], ascending = True)
        clust_props_df = clust_props_df[clust_props_df['Cost_Price'].notna()]
        
        fut_demand_list = clust_props_df['future_demand'].tolist()
        curr_price_list = clust_props_df['Tesco_Regular_Price'].tolist()
        curr_cost_list = clust_props_df['Cost_Price'].tolist()
        
        ## Include TPNs only with relevant data
        clust_tpn_list = [tpn for tpn in clust_tpn_list if tpn in clust_props_df['TPN'].tolist()]
        clust_tpn_list.sort()
        
        ##Get Simulated Price Data
        new_price_df = tpn_price_df[tpn_price_df['TPN'].isin(clust_tpn_list)][['TPN', 'Price_Ranges']]
        new_price_df = new_price_df.sort_values(['TPN'], ascending = True)
        new_price_df = new_price_df.set_index(['TPN'])
        new_price_dict = new_price_df.to_dict()
        
        price_bounds_df = tpn_price_df[tpn_price_df['TPN'].isin(clust_tpn_list)][['TPN', 'Price_Ranges']]
        
        #################
        ##### Get the Price and Cross-Price Elasticities matrix
        #################
        clust_coeff_list = []
        
        for tpn_iter in clust_tpn_list:
            
                coeff_iter_df = feat2assoc_map_df[feat2assoc_map_df['TPN'] == str(tpn_iter)]
                coeff_iter_df = coeff_iter_df[['assoc_tpn','coefMod']]
                coeff_iter_df.columns = ['assocCap_tpn','coefMod']
        
                clust_iter_df = clust_tpn_df.merge(coeff_iter_df, on = "assocCap_tpn", how = "left")
                clust_iter_df['assocCap_tpn'] = pd.to_numeric(clust_iter_df['assocCap_tpn'])
                clust_iter_df['coefMod'] = clust_iter_df['coefMod'].fillna(0)
                clust_iter_df = clust_iter_df[clust_iter_df['assocCap_tpn'].isin(clust_tpn_list)]
                
                clust_iter_df = clust_iter_df.sort_values(['assocCap_tpn'], ascending = True)
                
                clust_coeff_list.append(clust_iter_df['coefMod'])
        
        
        ##################
        ## get Section Mapping for all the TPNs
        ##################
        # #Add TPN Descriptions
        # sales_vol_file = "base_captain_flagged_22p09_23p08_all.xlsx"
        # sales_vol_df = pd.read_excel(os.path.join(Path(base_loc, sales_vol_loc, sales_vol_file)), sheet_name = "Sheet1")
        # tpn_mapping_df = sales_vol_df[['slad_tpn','dmat_dep_des','dmat_sec_des','dmat_grp_des','dmat_sgr_des']]
        
        # clust_tpn_map_df = tpn_mapping_df[tpn_mapping_df['slad_tpn'].isin(clust_tpn_list)]
        # clust_tpn_map_df = clust_tpn_map_df.merge(cluster_info_df_upd[['TPN','Assoc_TPN']], left_on = "slad_tpn", right_on = "Assoc_TPN", how = "left")
        # clust_tpn_map_df['Prim_Flag'] = np.where(clust_tpn_map_df['TPN'] == clust_tpn_map_df['Assoc_TPN'], 1, 0)
        
        # # Select products from Same Section
        # prim_clust_map_df = clust_tpn_map_df[clust_tpn_map_df['Prim_Flag']==1] 
        # prim_clust_map_df = prim_clust_map_df[['dmat_dep_des','dmat_sec_des']].drop_duplicates()
        # prim_clust_map_df = prim_clust_map_df.merge(clust_tpn_map_df[['slad_tpn','dmat_dep_des','dmat_sec_des','dmat_grp_des','dmat_sgr_des']], 
        #                                             on = ['dmat_dep_des','dmat_sec_des'], how = "left")
        
        # Select TOP Selling products for Optimization
        # prim_clust_map_df = clust_tpn_map_df[clust_tpn_map_df['Prim_Flag']==1] 
        # prim_clust_map_df = prim_clust_map_df[['TPN','dmat_dep_des','dmat_sec_des']].drop_duplicates()
        # prim_clust_map_df = prim_clust_map_df.merge(clust_tpn_map_df)
    
        #################
        #### Multiprocessing based optimization
        #################
        # import numpy as np
        # from multiprocessing import Pool
        # from joblib import Parallel, delayed
        # from joblib.parallel import ParallelBackend
        # backend = ParallelBackend(n_jobs=-1)  # use all available cores
        # from hyperopt import hp, space_eval, tpe, fmin
        
        # Define the constant matrices
        elast_matrix, demand_matrix, cost_matrix, price_matrix = clust_coeff_list, fut_demand_list, curr_cost_list, curr_price_list
        
        # Define the objective function
        def objective(params):
        
            #Define Price_Iter as Params by reading out values from Params    
            price_iter = [] 
            for i in range(len(list(new_price_dict['Price_Ranges'].values()))):
                variable = "X"+str(i)    
                price_iter.append(params[variable])
                        
            # Current Margin
            curr_result = np.sum((1+(np.dot(np.array(elast_matrix), (np.subtract(np.array(curr_price_list), np.array(curr_price_list))/ np.array(curr_price_list)).T))) * demand_matrix * np.array(np.subtract(curr_price_list, curr_cost_list)))
            
            ###Possibel Margin
            iter_result = np.sum((1+(np.dot(np.array(elast_matrix), (np.subtract(np.array(price_iter), np.array(curr_price_list))/ np.array(curr_price_list)).T))) * demand_matrix * np.array(np.subtract(price_iter, curr_cost_list)))
        
            ## Margin Improvements
            net_margin = curr_result - iter_result
            
            return net_margin
        
        # ### Warp the objective function
        # def objective_wrapped(params):
        #     y = objective(params)
        #     return y
        
        # # Wrap the objective function with joblib.delayed for parallel execution
        # objective_parallel = delayed(objective_wrapped)
        
        # Create a DiscreteSpace object
        def get_search_space(price_dict):
            search_space_comb = {}
            for i in range(len(list(price_dict['Price_Ranges'].values()))):
                variable = "X"+str(i)
                search_space_comb[variable] = hp.choice(variable, list(new_price_dict['Price_Ranges'].values())[i])
            return search_space_comb
        
        # Use the dynamic search space
        search_space = get_search_space(new_price_dict)
        
        ##Decalring number of evals
        n_evals = np.ceil(len(clust_tpn_list) ** 0.25) * 1000
        
        # try:
        # Optimize using Tree-structured Parzen Estimator (TPE)
        best = fmin(
            fn=objective,
            space=search_space,
            algo=tpe.suggest,
            max_evals=n_evals,
            verbose = True,
        )
    
        #Get Best Iter    
        best_iter = [] 
        for i in range(len(best)):
            variable = "X"+str(i)    
            best_index = best[variable]
            best_iter.append(list(new_price_dict['Price_Ranges'].values())[i][best_index])
                        
        ## Get Best Result - Margins
        curr_result = list((1+(np.dot(np.array(elast_matrix), (np.subtract(np.array(curr_price_list), np.array(curr_price_list))/ np.array(curr_price_list)).T))) * demand_matrix * np.array(np.subtract(curr_price_list, curr_cost_list)))
        net_best_result = list((1+(np.dot(np.array(elast_matrix), (np.subtract(np.array(best_iter), np.array(curr_price_list))/ np.array(curr_price_list)).T))) * demand_matrix * np.array(np.subtract(best_iter, curr_cost_list)))

        ## Get Best Result - Demand
        curr_result_demand = list((1+(np.dot(np.array(elast_matrix), (np.subtract(np.array(curr_price_list), np.array(curr_price_list))/ np.array(curr_price_list)).T))) * demand_matrix)
        net_best_result_demand = list((1+(np.dot(np.array(elast_matrix), (np.subtract(np.array(best_iter), np.array(curr_price_list))/ np.array(curr_price_list)).T))) * demand_matrix)
                
        #### Store Outputs in a df
        iter_pricing_result = pd.DataFrame(clust_tpn_list)
        iter_pricing_result.columns = ['TPN']
        iter_pricing_result = iter_pricing_result.sort_values(['TPN']) 
        iter_pricing_result['Cluster'] = clust_iter
        iter_pricing_result['Cost'] = curr_cost_list

        # As-Is vs To Be Scenarios - Price
        iter_pricing_result['AsIsPrice'] = curr_price_list
        iter_pricing_result['ToBePrice'] = best_iter

        # As-Is vs To Be Scenarios - Demand
        iter_pricing_result['AsIsDemand'] = curr_result_demand
        iter_pricing_result['ToBeDemand'] = net_best_result_demand

        # As-Is vs To Be Scenarios - Sales
        iter_pricing_result['AsIsSales'] = iter_pricing_result['AsIsDemand'] * iter_pricing_result['AsIsPrice']
        iter_pricing_result['ToBeSales'] = iter_pricing_result['ToBeDemand'] * iter_pricing_result['ToBePrice']

        # As-Is vs To Be Scenarios - Margin
        iter_pricing_result['AsIsMargin'] = curr_result
        iter_pricing_result['ToBeMargin'] = net_best_result
    
        pricing_opt_net_result.append(iter_pricing_result)
            
    except:
        
        print(f"Cluster failed as no cost price found: {clust_iter}")
        
        
final_bb_result = pd.concat(pricing_opt_net_result)

print("--- %s seconds ---" % (time.time() - start_time))

