'''This .py file contains all of the functions created for the recidivism analysis'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm


def create_dummy_range(lower, upper, risk_df, result, prefix):
    '''Adds columns of indicator variables to an existing dataframe for the desired range. 
            
        Params:
            lower (int): The lower bound for the range
            upper (int): The upper bound for the range
            risk_df (DataFrame): DataFrame containing all relevant information for risk rankings and other characteristics
            result (DataFrame): DataFrame that will be constructed to have groupings 
            prefix (String): column name (e.g., Months Supervised) to be followed by specified grouping
  
    '''
    
    col_name = prefix + ": " + str(lower) + '-' + str(upper)
    result[col_name] = risk_df[prefix].apply(lambda x: 1 if x in np.arange(lower, upper+1) else 0) # 1 if in age group, o.w. 0
    
    

def dummy_mlr(variable, risk_df):
    '''Creates a multiple linear regression object with Risk Ranking as the endogenous variable, and the desired variable as our exogenous variable, to determine the effect of the desired variable on Risk Ranking assignment.
    
    
    Params:
        variable (String): exogenous variable for the regression (e.g., Age)
        risk_df (DataFrame): DataFrame with all data from risk rankings, including grouped dummies
    
    Returns:
        mlr (statsmodels.discrete.discrete_model.MultinomialResultsWrapper): MLR object that contains our coefficients
    
    '''
    mlr = sm.MNLogit(endog = risk_df.loc[:, ['Risk Ranking']],
                 exog = risk_df.loc[:, [col for col in risk_df if variable in col]]).fit() 
    # note we use a list comprehension to obtain all of the columns related to variable (e.g., for Age, we get all groupings)
    return mlr



def chamberlain(risk_ranking, quantile, risk_df):
    '''Calculates the Chamberlain estimate of a desired quantile of a distribution of survival durations, the 95% Confidence Interval of that quantile estimate, and the estimated variance of the quantiles - returns all of these in a single list for ease of calculation with future analysis
    
    Params:
        risk_ranking (String): The Risk Ranking that we want to find the quantile for (Low, Moderate, or High)
        quantile: Desired quantile we wish to calculate (can be any valid percentile value)
        risk_df (Dataframe): DataFrame that contains the right censored data 
        
    Returns:
        A list containing each of the following values:
            estimate (float): The Chamberlain estimate of the quantile of the distribution of survival times for the given ranking 
            ci (tuple): The 95% Confidence Interval of estimate 
            var (float): The variance of estimate 
        
    
    '''
    
    
    X_l = risk_df.loc[risk_df['Risk Ranking'] == risk_ranking, 'Survival Time (Months)'].values
    N_l = len(X_l)
    l = 1.96 * np.sqrt(N_l * quantile * (1 - quantile))
    j = int(np.floor(N_l * quantile - l))
    k = int(np.ceil(N_l * quantile + l))
    
    order_stats = np.sort(X_l)
    
    X_j = order_stats[j-1] # we do j-1 because of Python's zero indexing 
    X_k = order_stats[k-1]
    
    estimate = (X_j + order_stats[j]) / 2 
    ci = (X_j, X_k) 
    var = (N_l * (X_k - X_j)**2) / (4 * 1.96**2)
    
    return [estimate, ci, var]

def mde(percentile, matrix):
    '''Calculates the coefficients (the estimate of the survival time) of our risk rankings for a given percentile using the minimum distance estimation
    
    
    Params:
        percentile (float): Percentile for which we want to calculate the coefficient
        matrix (DataFrame): DataFrame that contains all information needed for the MDE 
        
    Returns:
        None
        
        '''
        
    df = matrix.loc[matrix['Percentile'] == percentile]
    G = df[['Constant', 'Risk Ranking: High', 'Risk Ranking: Low', 'Risk Ranking: Moderate']].values
    Omega = np.diag(df['$\hat{p}$'].values / df['Variance'].values)
    pi_hat = df[['Estimate']].values
    
    theta_hat = np.linalg.inv(G.T@np.linalg.inv(Omega)@G)@(G.T@np.linalg.inv(Omega)@pi_hat)
    
    print(f'========= Quantile: {str(int(percentile*100)) + "th"} =========')
    print(f'MDE Coefficient for Constant: {theta_hat[0]}')
    print(f'MDE Coefficient for High Risk Ranking: {theta_hat[1]}')
    print(f'MDE Coefficient for Low Risk Ranking: {theta_hat[2]}')
    print(f'MDE Coefficient for Moderate Risk Ranking: {theta_hat[3]}')
    print()



def lifetable(df):
    
    ''' Creates a lifetable with right censoring from the original duration dataframe
    
        Params:
            df (DataFrame): Original dataframe (must contain duration at the very minimum)
        
        Returns: 
            lifetable (DataFrame): The right-censored life table
    
    '''
    
    df['D'] = df[['Survival Time (Months)']].notnull() # D is an indicator for observed duration
    df['Z'] = sample[['Survival Time (Months)']].apply(np.ceil) # Z is the survival time in months
    df['Z'].fillna(37, inplace = True) # right censoring at 37 months (observation ends at 36 months)
    df['Z'] = df['Z'].astype(int)
    
    grp = df.groupby('Z')
    lifetable = pd.DataFrame({"N_z": grp.size(), "Re-incarcerated": grp["D"].sum()})

    N = len(df)
    prior_count = lifetable["N_z"].cumsum().shift(1, fill_value=0)
    lifetable.insert(0, "At risk", N - prior_count) 

    lifetable["Censored"] = (lifetable["At risk"] - lifetable["At risk"].shift(-1, fill_value=0) - lifetable["Re-incarcerated"]) 

    lifetable.drop(['N_z'], axis=1, inplace = True)

    return lifetable



def kaplan_meier(lifetable):
    
    ''' Appends the Kaplan-Meier survival function and hazard function estimates to our lifetable 
    
        Params: 
            lifetable (DataFrame): a right-censored life table as generated by the lifetable(df) function
        
        ''' 
    
    lifetable['$\lambda(y)$'] = lifetable["Re-incarcerated"] / lifetable["At risk"] # hazard function estimate
    lifetable['$S(y)$'] = (1-lifetable['$\lambda(y)$']).cumprod() # survival function estimate
    lifetable.reset_index(inplace=True)
    
    

def ci_greenwood(lifetable):
    ''' Creates a 95% confidence interval of the Kaplan-Meier estimate
    
        Params:
            lifetable (Dataframe): right-censored life table with the Kaplan-Meier estimate generated by kaplan_meier(lifetable)
        
        Returns:
            (lower_ci, upper_ci) (tuple[float, float]): the 95% CI of the Kaplan-Meier estimate
            
    '''

    greenwood = df['$S(y)$'] * ((df['Re-incarcerated'] / (df['At risk'] * (df['At risk'] - df['Re-incarcerated']))).cumsum())**0.5
    upper_ci = df['$S(y)$'] + (1.96 * greenwood)
    lower_ci = df['$S(y)$'] - (1.96 * greenwood)
    return (lower_ci, upper_ci)



def plot_survival_hazard(df1, name1, df2, name2):
    
    ''' Plots the estimated survival functions with 95% CIs and the first quartile of the survival function on one graph, and the estimated hazard function on another graph, for the two given dataframes (e.g., comparing Male vs. Female)
    
        Params:
            df1: right-censored life table with Kaplan-Meier estimate from kaplan_meier(lifetable) for first group (e.g., Female)
            name1: Name of first group to be used in titles and legend lables
            df2: right-censored life table with Kaplan-Meier estimate from kaplan_meier(lifetable) for second group (e.g., Male)
            name2: Name of second group to be used in titles and legend lables
    '''

    fig_survival, sbp = plt.subplots(ncols=2, figsize=(8, 4))
    ax0 = sbp[0] 
    ax1 = sbp[1] 
    fig_survival.suptitle(f'Recidivism of {name1} vs. {name2}', fontsize=14)

    color = '#3B7EA1'                                                                        # Founder's Rock

    df1_quartile_1 = df1.loc[df1['$S(y)$'] <= 0.75].iloc[0]['Z']
    df2_quartile_1 = df2.loc[df2['$S(y)$'] <= 0.75].iloc[0]['Z']

    # Survival function
    ax0.set_xlabel(r'Months since release')
    ax0.set_ylabel(r'Surival function, $S(y)$')
    s = ax0.step(df1['Z'], df1['$S(y)$'], alpha = 0.75, where = 'post', label = f'{name1} $S(y)$', color = 'blue')
    s = ax0.step(df1['Z'], ci_greenwood(df1)[0], alpha = 0.75, where = 'post', color = 'green', linestyle = '--')
    s = ax0.step(df1['Z'], ci_greenwood(df1)[1], alpha = 0.75,
                 where = 'post', color = 'green', label = '95% CI', linestyle = '--')

    s = ax0.step(df2['Z'], df2['$S(y)$'], alpha = 0.75, where = 'post', label = f'{name2} $S(y)$', color = 'orange')
    s = ax0.step(df2['Z'], ci_greenwood(df2)[0], alpha = 0.75, where = 'post', color = 'red', linestyle = '--')
    s = ax0.step(df2['Z'], ci_greenwood(df2)[1], alpha = 0.75, 
                 where = 'post', color = 'red', label = '95% CI', linestyle = '--')

    s = ax0.axvline(df1_quartile_1, label = f'{name1} 1st Quartile', linestyle = ":", color = 'blue')
    s = ax0.axvline(df2_quartile_1, label = f'{name2} 1st Quartile', linestyle = ":", color = 'orange')

    ax0.tick_params(axis='y')
    ax0.set_xticks([1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34, 37])
    ax0.set_yticks([1, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50])
    ax0.set_yticklabels(['1.00', '0.95', '0.90', '0.85', '0.80', '0.75', '0.70', '0.65', '0.60', '0.55', '0.50'])
    ax0.legend(fontsize = 7)

    # Hazard function
    ax1.set_xlabel(r'Months since release')
    ax1.set_ylabel(r'Hazard function, $\lambda(y)$')
    s = ax1.step(df1['Z'], df1['$\lambda(y)$'], alpha = 0.75, where = 'post', label = f'{name1} $\lambda(y)$')
    s = ax1.step(df2['Z'], df2['$\lambda(y)$'], alpha = 0.75, where = 'post', label = f'{name2} $\lambda(y)$')

    ax1.tick_params(axis='y')
    ax1.set_xticks([1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34, 37])
    ax1.legend()

    fig_survival.tight_layout() 