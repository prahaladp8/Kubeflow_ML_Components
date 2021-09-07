import numpy as np 
import pandas as pd 
from scipy.stats import chi2_contingency
from statsmodels.tools.tools import add_constant
from statsmodels.stats.outliers_influence import variance_inflation_factor


def compute_csi_numeric(series1,series2):
    #series1_endIndex = series1.count()
    bin_range = [0.0, 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
    deciles = series1.quantile(bin_range)
    deciles[0.0] = min(series1.min(), series2.min())
    deciles[1.0] = max(series1.max(), series2.max())
    #print(f'bin_range = {bin_range}')

    df1 = pd.DataFrame(series1)
    df2 = pd.DataFrame(series2)    

    df1_bins = pd.Series(np.histogram(df1,bins=deciles)[0],name='expected')
    df2_bins = pd.Series(np.histogram(df2,bins=deciles)[0],name='actual')

    csi_df = pd.DataFrame([df1_bins,df2_bins])
    csi_df = csi_df.T
    
    #print(csi_df)        
    csi_df['expected'] = csi_df['expected'] / (csi_df['expected'].sum())
    csi_df['actual'] = csi_df['actual'] / (csi_df['actual'].sum())
    csi_df['diff'] = csi_df['actual'] - csi_df['expected']
    csi_df['psi'] = (csi_df['diff'])*np.log(csi_df['actual'] / csi_df['expected'])
    csi_df['psi'] = csi_df['psi'].replace([np.nan, np.inf, -np.inf],0)

    return csi_df['psi'].sum()
    #print(df.head(5))

def compute_csi_categorical(series1,series2):
    
    df1_value_count = series1.value_counts()
    df2_value_count = series2.value_counts()

    df1_indexes = df1_value_count.index.tolist()
    df2_indexes = df2_value_count.index.tolist()

    categorical_values_list = list(set(df1_indexes+df2_indexes))        

    val = {"Values":categorical_values_list}
    df = pd.DataFrame(val)
    df['Expected'] = 0 
    df['Actual'] = 0 

    for categorical_value in categorical_values_list:
        if categorical_value in df1_indexes:
            df.loc[(df.Values == categorical_value),'Expected'] = df1_value_count.loc[categorical_value]
        if categorical_value in df2_indexes:
            df.loc[(df.Values == categorical_value),'Actual'] = df2_value_count.loc[categorical_value]

    df['Expected'] = df['Expected'] / df['Expected'].sum()
    df['Actual'] = df['Actual'] / df['Actual'].sum()

    df['Difference'] =  df['Actual'] - df['Expected']

    df['CSI'] =  (df['Difference']) * np.log(df['Actual'] / df['Expected']) 
    df['CSI'] = df['CSI'].replace([np.nan, np.inf, -np.inf],0)

    return df.CSI.sum()

def compute_csi(training_set = None, oot_set=None, csi_threshold=0.25,
                write_output=False,output_path='', final_feature_set=None):
    
    if final_feature_set is None:
        final_feature_set = training_set.columns.to_list()
        
    csi_dict = { 'Feature' : [], 'CSI' : [] }
    for feature in final_feature_set:
        if(feature in ["Unnamed: 0","emp_title","title"]):
            continue
        #print(feature)
        csi_dict['Feature'].append(feature)
        csi_value = 0

        if training_set.dtypes[feature] in ['int64','float64']:
            csi_value = compute_csi_numeric(training_set[feature],oot_set[feature])
        if training_set.dtypes[feature] in ['object']:
            csi_value = compute_csi_categorical(training_set[feature],oot_set[feature])                                        
        csi_dict['CSI'].append(round(csi_value,4))
    csi_df = pd.DataFrame(csi_dict) 
    if write_output:
        csi_df.to_csv(output_path,index=False)
    return csi_df, final_feature_set


def compute_vif(dataset=None, vif_threshold=10,
                write_output=False, output_path='', final_feature_set=None):
    if final_feature_set is None:
        final_feature_set=dataset.columns.to_list()
    
    feature_list = []
    for feature in final_feature_set:
        if(dataset.dtypes[feature] not in ['int64','float64']):
            continue
        else:
            feature_list.append(feature)        

    #self.log("Computing VIFs to check for Multicolinearity between {} Features".format(feature_list))
    print("Computing VIFs to check for Multicolinearity between {} Features".format(feature_list))
    vif_ds = add_constant(dataset[feature_list])
    vifs = pd.Series([variance_inflation_factor(vif_ds.values, i) for i in range(vif_ds.shape[1])], index=vif_ds.columns)        
    #self.log(vifs)
    vifs = vifs.sort_values(ascending=False)
    #self.convert_df_to_html(vifs.to_frame(),self.pipeline_configuration['reports_directory'],'Multicolinearity_Report',hide_index=False)
    eliminated_features = vifs[vifs >= vif_threshold].index.tolist()
    #self.log("Removing features with VIFs greater than {}".format(ExecutionStepInputs.VIF_THRESHOLD))
    print("Removing features with VIFs greater than {}".format(vif_threshold))
    vifs = vifs[vifs <= vif_threshold]
    #self.log("Eliminated following feature : {}".format(eliminated_features))
    print("Eliminated following feature : {}".format(eliminated_features))
    #vifs.to_csv("Reports/Multicolinearity_Report.csv",index=True)
    if write_output:
        vifs.to_csv(output_path, index=False)

    return vifs, final_feature_set, eliminated_features

def build_correlation_matrix(dataset, correlation_threshold=0.7,
                            write_output=False,output_path='', final_feature_set=None, ivs=None):
    if final_feature_set is None:
        final_feature_set = dataset.columns.to_list()
    numeric_features = dataset[final_feature_set].select_dtypes(include=['number'])
    correlation_matrix = numeric_features.corr()
    correlation_matrix.index.name = 'numeric_features_list'

    correlation_matrix.reset_index(inplace=True)

    cols = correlation_matrix.columns
    eliminated_features = set()

    for feature1 in cols:
        for feature2 in cols:
            if (feature1 == feature2) or (feature1 == 'numeric_features_list') or (feature2 == 'numeric_features_list'):
                continue
            correlation = correlation_matrix.loc[correlation_matrix['numeric_features_list'] == feature1 , feature2 ].values[0]
            if(correlation > correlation_threshold):
                #Picking the feature with the higher IV of the 2 feature
                if ivs[feature1] > ivs[feature2]:
                    eliminated_features.add(feature2)
                else:
                    eliminated_features.add(feature1)
                
    #self.log('Correlation > Threshold {} hence eleminating {}'.format(ExecutionStepInputs.CORRELATION_THRESHOLD,str(eliminated_features)))
    #print(eliminated_features)
    print('Correlation > Threshold {} hence eleminating {}'.format(correlation_threshold,str(eliminated_features)))

    #self.log("Generating Correlation Report")
    print("Generating Correlation Report")
    #self.convert_df_to_html(correlation_matrix,self.pipeline_configuration['reports_directory'],'Correlation_Report')
    #self.log(correlation_matrix)
    if write_output:
        correlation_matrix.to_csv(output_path, index=False)
    
    return correlation_matrix, eliminated_features, final_feature_set


def get_variable_iv_score(feature_name,dataset, is_numeric=False):

    if (is_numeric):
        qlist = [np.round(i*0.1,1) for i in range(11)]
        quart = dataset[feature_name].quantile(qlist)
        temp_percentile_binning_train, feat_bins = pd.qcut(dataset[feature_name], 5 , precision=5, retbins=True, duplicates='drop' )
        #self.percentile_bin_ranges[feature_name] = feat_bins            
    if is_numeric: 
        #check shape of the below 2
        contigency= pd.crosstab(temp_percentile_binning_train,dataset['target'],dropna=True)
    else:
        contigency= pd.crosstab(dataset[feature_name],dataset['target'],dropna=True)
    chi_sqaure_data = {'Feature_Name':[],'WOE':[],'IV':[]}        
#     if ('0' not in contigency.columns) and ('1' not in contigency.columns):
#         print(feature_name)
#         return 0
    #print(contigency)
    #contigency = contigency.T
    good_count = contigency[0].sum()
    bad_count = contigency[1].sum()
    contigency["Percentage_Good"] = (contigency[0]/good_count)
    contigency["Percentage_Bad"] = (contigency[1]/bad_count)
    contigency["woe"] = np.log(contigency["Percentage_Good"]/contigency["Percentage_Bad"])
    contigency["woe"] = contigency["woe"].replace([np.nan, np.inf, -np.inf],0)
    contigency["IV"] = (contigency["Percentage_Good"] - contigency["Percentage_Bad"]) * contigency["woe"]
    #print("{} IV {}".format(feature_name,contigency['IV'].sum()))
    return contigency['IV'].sum()


def filter_by_iv(dataset=None, iv_threshold=0.02,
             write_output=False,output_path='', final_feature_set=None):
    #self.log("Computing IV scores for all independent variables - ")
    print("Computing IV scores for all independent variables - ")
    feature_iv = {"Feature":[],"IV":[]}
    if final_feature_set is None:
        final_feature_set = dataset.columns.to_list()

    numeric_features = dataset.select_dtypes(include=['number'])

    for feature in final_feature_set:
        feature_iv["Feature"].append(feature)
        iv = get_variable_iv_score(feature,dataset,feature in numeric_features)
        feature_iv["IV"].append(iv)   

    #iv_pd["IV"] = iv_pd["IV"].replace([np.nan, np.inf, -np.inf],0)
    iv_pd = pd.DataFrame(feature_iv)
    ivs = dict(iv_pd.values)
    iv_pd = iv_pd.sort_values('IV',ascending=False)


    iv_filter = iv_pd[iv_pd["IV"]<=iv_threshold]["Feature"].tolist()
    #Filter out items based on iv < = 0.02 here & add to list.
    #print(iv_filter)
    #iv_pd.to_csv('Reports/IV_Report.csv',index=True)
    #self.convert_df_to_html(iv_pd,self.pipeline_configuration['reports_directory'],'IV_Report',True)
    #self.log('{}'.format(iv_pd))
    #self.log("Removing following features are they are below the IV Threshold of {}".format(ExecutionStepInputs.IV_THRESHOLD))        
    print("Removing following features are they are below the IV Threshold of {}".format(iv_threshold))
    #self.log(iv_filter)
    print(iv_filter)
    #self.convert_df_to_html(iv_filter,self.pipeline_configuration[''],'IV_Report')        
    for feature in iv_filter:
        final_feature_set.remove(feature)

    if write_output:
        iv_pd.to_csv(output_path, index=False)

    return iv_pd, final_feature_set


def getCramerv(c,n,contigency):
        phi2 = c/n
        r,k = contigency.shape
        phi2corr = max(0.0, phi2 - (((k-1)*(r-1))/(n-1)))    
        rcorr = r - ((r-1)**2)/(n-1)
        kcorr = k - ((k-1)**2)/(n-1)
        cramers_v =  np.sqrt(phi2corr / min( (kcorr-1), (rcorr-1)))
        return cramers_v

def getChiSquare(dataset, cramersv_threshold=0.5,
             write_output=False,output_path='', cardinality_threshold=15, dependent_variable=None):
    
    final_feature_set = dataset.columns.to_list()
    chi_sqaure_data = {'Feature_Name':[],'Chi-Square':[],'P-Value':[],'dof':[],'Significant':[],'CramersV':[]}
    alpha=0.05        
    categoricals = dataset[final_feature_set].select_dtypes(include=['object','category']).columns
    eliminated_features = []

    for feature in categoricals:
        if feature == dependent_variable:
            continue
        if(len(dataset[feature].unique()) > 15):   #Eliminating features with cardinality > 15
            eliminated_features.append(feature)
            continue

        contigency= pd.crosstab(dataset[dependent_variable],dataset[feature])
        c, p, dof, expected = chi2_contingency(contigency)
        n =  sum(contigency.sum())
        chi_sqaure_data["Feature_Name"].append(feature)
        chi_sqaure_data["Chi-Square"].append(c)
        chi_sqaure_data["P-Value"].append(p)
        chi_sqaure_data["dof"].append(dof)            
        chi_sqaure_data['CramersV'].append(getCramerv(c,n,contigency))
    chi_sqaure_pd = pd.DataFrame(chi_sqaure_data,columns=['Feature_Name','Chi-Square','P-Value','dof','CramersV'])

    #self.convert_df_to_html(chi_sqaure_pd,"","CramersV")
    cramersv_filter = chi_sqaure_pd[chi_sqaure_pd["CramersV"]<=cramersv_threshold]["Feature_Name"].tolist()

    #self.log('Eliminating following variables as CramersV below Threshold {}'.format(ExecutionStepInputs.CRAMERSV_THRESHOLD))  
    #self.log(cramersv_filter)      
    print('Eliminating following variables as CramersV below Threshold {}'.format(cramersv_threshold)) 
    print(cramersv_filter)

    # self.log('Eliminating following variables as they have cardinality > 15 {}'.format(eliminated_features))
    # self.log(eliminated_features)
    print('Eliminating following variables as they have cardinality > 15 {}'.format(eliminated_features))
    print(eliminated_features)

    eliminated_features.extend(cramersv_filter)
    for feature in eliminated_features:
        final_feature_set.remove(feature)

    if write_output:
        chi_sqaure_pd.to_csv(output_path,index=False)

    return chi_sqaure_pd, final_feature_set