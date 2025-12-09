# ======================================================================================================================================================

# Code for the article Verdú, Carchano (2024). Oportunitats d'Arbitratge a les Ampliacions de Capital a la Comunitat Valenciana i Catañunya.
# Revista Econòmica de Catalunya. 90, 106.112.

# ======================================================================================================================================================

# To execute this code, you must have file ‘Verdu_Carchano_2025_Data.csv’ available in ‘https://doi.org/10.5281/zenodo.17871082’.

# Once the data is available in the same directory, you only need to execute the code to obtain the results shown in the article.

# The article can be found at: https://hdl.handle.net/10550/102650

# ======================================================================================================================================================

#################### LIBRARIES TO USE ####################

import numpy as np                      # Allows to work with Series.
import pandas as pd						# Allows to organize the Data.
import matplotlib.pyplot as plt         # Allows to work with graphs.
import statsmodels.formula.api as smf	# Allows to estimate the Econometrical models.

#################### START OF COMPLEMENTARY FUNCTIONS ####################

def calculate_stats(data, var_name='NET'):
    """
    Calculate descriptive statistics and hypothesis tests
    """
    from scipy import stats
    
    values = data[var_name].dropna()
    n = len(values)
    
    if n == 0:
        return {
            'N': 0,
            'Mean': np.nan,
            'Std': np.nan,
            'Median': np.nan,
            'T-test p-value': np.nan,
            'Wilcoxon p-value': np.nan
        }
    
    mean = values.mean()
    std = values.std()
    median = values.median()
    
    # T-test (testing if mean is different from 0)
    t_stat, t_pvalue = stats.ttest_1samp(values, 0)
    
    # Wilcoxon signed-rank test (testing if median is different from 0)
    try:
        wilcoxon_stat, wilcoxon_pvalue = stats.wilcoxon(values)
    except:
        wilcoxon_pvalue = np.nan
    
    return {
        'N': n,
        'Mean': mean,
        'Std': std,
        'Median': median,
        'T-test p-value': t_pvalue,
        'Wilcoxon p-value': wilcoxon_pvalue
    }

#################### END OF COMPLEMENTARY FUNCTIONS ####################

#################### START OF THE CODE ####################

DAT = pd.read_csv('Verdu_Carchano_2025_Data.csv', delimiter = ';') # Open the data file.

### Removing of missing observations ###

DAT = DAT[DAT['NET'] != '-']
DAT = DAT.reset_index()
DAT = DAT.drop('index', axis = 1)

### Initialisation of Relevant Variables

NET, ARB = [], []
IBE, MCN, MAB, DIN, INS, DIL = [], [], [], [], [], []
BAS, CIC, CNC, FIN, IND, INM, SAL, TEC, UTI = [], [], [], [], [], [], [], [], []
OUT = []

### Data extraction and Adjustment ###
     # To ensure data readability, commas are replaced by dots in quantitative variables with decimal numbers.

for n in range(0, len(DAT)):

	NET.append(float(DAT['NET'][n]))
	ARB.append(DAT['ARB'][n])
	IBE.append(DAT['IBEX'][n])
	MCN.append(DAT['MC'][n])
	MAB.append(DAT['MAB'][n])
	DIN.append(DAT['DIN'][n])
	INS.append(DAT['INS'][n])
	DIL.append(float(DAT['DIL'][n]))
	BAS.append(DAT['BAS'][n])
	CIC.append(DAT['CIC'][n])
	CNC.append(DAT['CNC'][n])
	FIN.append(DAT['FIN'][n])
	IND.append(DAT['IND'][n])
	INM.append(DAT['INM'][n])
	SAL.append(DAT['SAL'][n])
	TEC.append(DAT['TEC'][n])
	UTI.append(DAT['UTI'][n])
	OUT.append(DAT['OUT'][n])

### Generation of a Dictionary with the complete Sample ###

DATA_O = {'NET': NET, 'ARB': ARB, 'DIL': DIL, 'IBE': IBE, 'MCN': MCN, 'MAB': MAB, 'DIN': DIN, 'INS': INS
,'BAS': BAS, 'CIC': CIC, 'CNC': CNC, 'FIN': FIN, 'IND': IND, 'INM': INM, 'SAL': SAL, 'TEC': TEC, 'UTI': UTI
, 'OUT': OUT}
DATA_O = pd.DataFrame(DATA_O)

DATA_nO = DATA_O[DATA_O['OUT'] == 0] #Removing the atypical returns.

### Descriptive Statistics and Statistical Tests ###

## Calculate statistics for different subsamples ##

print("=" * 80)
print("DESCRIPTIVE STATISTICS AND HYPOTHESIS TESTS")
print("=" * 80)

## Total Sample ##

stats_total = calculate_stats(DATA_nO, 'NET')

## Dilutive Equity Offerings ##

data_dil_high = DATA_nO[DATA_nO['DIL'] >= 0.5]
stats_dil_high = calculate_stats(data_dil_high, 'NET')

## Non-Dilutive Equity Offerings ##

data_dil_low = DATA_nO[DATA_nO['DIL'] < 0.5]
stats_dil_low = calculate_stats(data_dil_low, 'NET')

## Dinerary Equity Offerings ##

data_din_1 = DATA_nO[DATA_nO['DIN'] == 1]
stats_din_1 = calculate_stats(data_din_1, 'NET')

## Non-Dinerary Equity Offerings ##

data_din_0 = DATA_nO[DATA_nO['DIN'] == 0]
stats_din_0 = calculate_stats(data_din_0, 'NET')

print("\n" + "=" * 80)

## Summary table ##
summary_stats = pd.DataFrame({
    'Total Sample': stats_total,
    'DIL >= 0.5': stats_dil_high,
    'DIL < 0.5': stats_dil_low,
    'DIN = 1': stats_din_1,
    'DIN = 0': stats_din_0
}).T

print("\nSummary Table:")
print(summary_stats)
print("\n" + "=" * 80)

### Econometric Models Estimation ###

## Formulation of the Models to be Estimated ##

FOR_nO = 'NET ~ DIL * DIN + DIN'	# Analysis of returns.
FOR_O = 'ARB ~ DIL * DIN + DIN'		# Analysis of arbitrage.

## OLS Model Estimation - Returns ##

mod_o = smf.ols(FOR_nO, DATA_nO)
res_o = mod_o.fit(cov_type = 'HC1') #White Robustness.

print('##### Resultados del Modelo MCO. #####')
print(res_o.summary())

## Logit Model Estimation - Arbitrage ##

mod_l = smf.logit(FOR_O, DATA_O)
res_l = mod_l.fit(cov_type = 'HC1') #White robustness.

print('##### Resultados del Modelo LOGIT. #####')
print(res_l.summary())

## Probit Model Estimation - Arbitrage ##

mod_p = smf.probit(FOR_O, DATA_O)
res_p = mod_p.fit(cov_type = 'HC1') #White robustness.

print('##### Resultados del Modelo PROBIT. #####')
print(res_p.summary())

### Histogram for NET ###

plt.figure(figsize=(10, 6))

## Create a copy and map values > 1 to 1.05 (to fall into a ">1" bin) ##

NET_hist = DATA_O['NET'].apply(lambda x: 1.05 if x > 1 else x)

bins = np.arange(-1.1, 1.2, 0.1)  # Now includes a bin from 1.0 to 1.1
n, bins_edges, patches = plt.hist(NET_hist, bins=bins, edgecolor='black')

## Color Bars - Red for Negative, Green for Positive ##

for i, patch in enumerate(patches):
    if bins_edges[i] >= 0:
        patch.set_facecolor('green')
    else:
        patch.set_facecolor('red')

plt.xlabel('NET (%)')
plt.ylabel('Frequency')
plt.title('Histograma de Rendiments Nets')
plt.grid(True, alpha=0.3)
plt.axvline(x=0, color='black', linestyle='--', linewidth=1)

## Custom x-axis: replace "1.0" tick with ">1" ##

ax = plt.gca()
xticks = np.arange(-1, 1.2, 0.2)
xticklabels = [f'{x:.1f}' for x in xticks[:-1]] + ['>1']
ax.set_xticks(xticks)
ax.set_xticklabels(xticklabels)

plt.show()

#################### END OF THE CODE ####################
