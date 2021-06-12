

#Loading the libraries required
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import datetime as dt


#Loading the data set and assigning a dataframe
df_nat= pd.read_csv("C:\\Users\\anupam.b.kumar.singh\\Documents\\Personal_Files\\Predictive Analysis1\\Assignment PCA&Clustering\\Country-data.csv")
df_nat.head(10)

#check for null values
df_nat.isnull().sum(axis=0)


#Getting info about the data set
df_nat.info()


#dataframe shape
df_nat.shape


#changig the Percetage numbers to actuall figures
df_nat['exports']=df_nat['exports'] * df_nat['gdpp'] /100
df_nat['health']=df_nat['health'] * df_nat['gdpp'] /100
df_nat['imports']=df_nat['imports'] * df_nat['gdpp'] /100

#assigning the dataframe to a new df with no country
df_nat2=df_nat.drop('country', axis=1)
df_natx=df_nat2
df_naty=df_nat2
df_nat2.head(10)

#Standardising the Data set
from sklearn.preprocessing import StandardScaler
standard_scaler = StandardScaler()
df_naty = standard_scaler.fit_transform(df_naty)

df_naty=pd.DataFrame(df_naty)
df_naty.columns=['child_mort', 'exports', 'health', 'imports', 'income', 'inflation',
       'life_expec', 'total_fer', 'gdpp']
df_naty.head(10)
df_nat=pd.concat([df_nat.country,df_naty], axis=1)
df_nat.head()

fig = plt.figure(figsize = (12,8))
sns.heatmap(df_naty.corr(), annot=True)
plt.show()

#Improting the PCA module
from sklearn.decomposition import PCA
pca = PCA(svd_solver='randomized', random_state=42)
df_pca = pca.fit_transform(df_naty)
pca.components_
df_pca.shape

#Making the screeplot - plotting the cumulative variance against the number of components
get_ipython().run_line_magic('matplotlib', 'inline')
fig = plt.figure(figsize = (12,8))
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.title('Scree Plot')
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
plt.show()

pca.explained_variance_ratio_
pca_final = PCA(svd_solver='randomized', random_state=42, n_components=4)
df_pca = pca_final.fit_transform(df_pca)
df_pca.shape
pca_final.components_
colnames = list(df_naty.columns)
pcs_df = pd.DataFrame({'PC1':pca_final.components_[0],'PC2':pca_final.components_[1],
                       'PC3':pca_final.components_[2],'PC4':pca_final.components_[3],'Feature':colnames})
pcs_df.head(10)

get_ipython().run_line_magic('matplotlib', 'inline')
fig = plt.figure(figsize = (8,6))
plt.scatter(pcs_df.PC1, pcs_df.PC2)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
for i, txt in enumerate(pcs_df.Feature):
    plt.annotate(txt, (pcs_df.PC1[i],pcs_df.PC2[i]))
plt.tight_layout()
plt.show()

#plotting the correlation matrix
get_ipython().run_line_magic('matplotlib', 'inline')
plt.figure(figsize = (10,8))
sns.heatmap(pcs_df.corr(),annot = True)

get_ipython().run_line_magic('matplotlib', 'inline')
fig = plt.figure(figsize = (8,6))
plt.scatter(df_pca[:,0], df_pca[:,1])
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.tight_layout()
plt.show()

#converting components(eigen vectors) into a dataframe
colnames = list(df_naty.columns)
pcs_df = pd.DataFrame({'PC1':pca.components_[0],'PC2':pca.components_[1],
                         'PC3':pca.components_[2],'PC4':pca.components_[3],'Feature':colnames})
pcs_df.head(10)

#clustering
from sklearn.neighbors import NearestNeighbors
from random import sample
from numpy.random import uniform
import numpy as np
from math import isnan
 
def hopkins(X):
    d = X.shape[1]
    #d = len(vars) # columns
    n = len(X) # rows
    m = int(0.1 * n) 
    nbrs = NearestNeighbors(n_neighbors=1).fit(X.values)
 
    rand_X = sample(range(0, n, 1), m)
 
    ujd = []
    wjd = []
    for j in range(0, m):
        u_dist, _ = nbrs.kneighbors(uniform(np.amin(X,axis=0),np.amax(X,axis=0),d).reshape(1, -1), 2, return_distance=True)
        ujd.append(u_dist[0][1])
        w_dist, _ = nbrs.kneighbors(X.iloc[rand_X[j]].values.reshape(1, -1), 2, return_distance=True)
        wjd.append(w_dist[0][1])
 
    H = sum(ujd) / (sum(ujd) + sum(wjd))
    if isnan(H):
        print(ujd, wjd)
        H = 0
 
    return H


pcs_df = pd.DataFrame({'PC1':df_pca[:,0],'PC2':df_pca[:,1], 'PC3':df_pca[:,2],'PC4':df_pca[:,3]})
pcs_df.head(10)
hopkins(pcs_df)

#Importing Libraries
import pandas as pd

# For Visualisation
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

# To Scale our data
from sklearn.preprocessing import scale

# To perform KMeans clustering 
from sklearn.cluster import KMeans

# To perform Hierarchical clustering
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import cut_tree

# Kmeans with K=5
model_clus5 = KMeans(n_clusters = 4, max_iter=50)
model_clus5.fit(pcs_df)


from sklearn.metrics import silhouette_score
sse_ = []
for k in range(2, 10):
    kmeans = KMeans(n_clusters=k).fit(pcs_df)
    sse_.append([k, silhouette_score(pcs_df, kmeans.labels_)])

plt.plot(pd.DataFrame(sse_)[0], pd.DataFrame(sse_)[1])
plt.show()

# sum of squared distances
ssd = []
for num_clusters in list(range(1,21)):
    model_clus = KMeans(n_clusters = num_clusters, max_iter=50)
    model_clus.fit(pcs_df)
    ssd.append(model_clus.inertia_)

plt.plot(ssd)

# Kmeans with K=4
model_clus4 = KMeans(n_clusters = 4, max_iter=50, random_state=0)
model_clus4.fit(pcs_df)

model_clus4.labels_


pcs_df.info()
df1 = pd.concat([df_nat,pcs_df], axis=1)

df1.head(10)


final_df = pd.concat([df1, pd.Series(model_clus4.labels_).rename('ClusterID')], axis=1)

get_ipython().run_line_magic('matplotlib', 'inline')
fig = plt.figure(figsize = (8,6))
sns.scatterplot(final_df['PC1'], final_df['PC2'], hue=final_df['ClusterID'], palette="Dark2")
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.tight_layout()
plt.show()

final_df[(final_df['ClusterID']==3)]

pcs_df2=final_df

km_clust_PC1 = pd.DataFrame(final_df.groupby(["ClusterID"]).PC1.mean())
km_clust_PC2 = pd.DataFrame(final_df.groupby(["ClusterID"]).PC2.mean())
km_clust_PC3 = pd.DataFrame(final_df.groupby(["ClusterID"]).PC3.mean())
km_clust_PC4 = pd.DataFrame(final_df.groupby(["ClusterID"]).PC4.mean())
km_clust_child_mort = pd.DataFrame(final_df.groupby(["ClusterID"]).child_mort.mean())
km_clust_health = pd.DataFrame(final_df.groupby(["ClusterID"]).health.mean())
km_clust_total_fer =  pd.DataFrame(final_df.groupby(["ClusterID"]).total_fer.mean())
km_clust_income = pd.DataFrame(final_df.groupby(["ClusterID"]).income.mean())
km_clust_life_expec = pd.DataFrame(final_df.groupby(["ClusterID"]).life_expec.mean())


df1 = pd.concat([pd.Series([0,1,2,3]), km_clust_PC1, km_clust_PC2,km_clust_PC3,km_clust_PC4], axis=1)
df1.columns = ["ClusterID", "PC1_mean", "PC2_mean","PC3_mean","PC4_mean"]
df1


df2 = pd.concat([pd.Series([0,1,2,3]),km_clust_child_mort,km_clust_health,km_clust_total_fer,km_clust_income,
                 km_clust_life_expec], axis=1)
df2.columns = ["ClusterID","child_mort_mean","health_mean","total_fer_mean","income_mean","life_expec_mean"]
df2

sns.barplot(x=df2.ClusterID, y=df2.child_mort_mean)
sns.barplot(x=df2.ClusterID, y=df2.health_mean)
sns.barplot(x=df2.ClusterID, y=df2.total_fer_mean)
sns.barplot(x=df2.ClusterID, y=df2.income_mean)
sns.barplot(x=df2.ClusterID, y=df2.life_expec_mean)

df4=final_df[(final_df.ClusterID==1)]

len(df4)
df4.head(10)
df4[df4.country=='Zambia']


dff = pd.concat([df_nat,pcs_df], axis=1)


# heirarchical clustering
mergings = linkage(pcs_df, method = "single", metric='euclidean')
dendrogram(mergings)
plt.show()


plt.figure(figsize=(15,10))
mergings = linkage(pcs_df, method = "complete", metric='euclidean')
dendrogram(mergings)
plt.show()

clusterCut = pd.Series(cut_tree(mergings, n_clusters = 5).reshape(-1,))
#Add the cluster Id to the df
df_hc = pd.concat([dff, clusterCut.rename('ClusterID')], axis=1)

#Analyse the scatter plot
get_ipython().run_line_magic('matplotlib', 'inline')
fig = plt.figure(figsize = (8,6))
sns.scatterplot(df_hc['PC1'], df_hc['PC2'], hue=df_hc['ClusterID'], palette="Dark2_r")
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.tight_layout()
plt.show()

len(final_df[(df_hc['ClusterID']==0)])

len(final_df[(df_hc['ClusterID']==1)])

len(final_df[(df_hc['ClusterID']==2)])

len(final_df[(df_hc['ClusterID']==3)])

df_hc.ClusterID.unique()

hc_clst_PC1 = pd.DataFrame(df_hc.groupby(["ClusterID"]).PC1.mean())
hc_clst_PC2 = pd.DataFrame(df_hc.groupby(["ClusterID"]).PC2.mean())
hc_clst_PC3 = pd.DataFrame(df_hc.groupby(["ClusterID"]).PC3.mean())
hc_clst_PC4 = pd.DataFrame(df_hc.groupby(["ClusterID"]).PC4.mean())
hc_clst_child_mort = pd.DataFrame(df_hc.groupby(["ClusterID"]).child_mort.mean())
hc_clst_health = pd.DataFrame(df_hc.groupby(["ClusterID"]).health.mean())
hc_clst_total_fer =  pd.DataFrame(df_hc.groupby(["ClusterID"]).total_fer.mean())
hc_clst_income = pd.DataFrame(df_hc.groupby(["ClusterID"]).income.mean())
hc_clst_life_expec = pd.DataFrame(df_hc.groupby(["ClusterID"]).life_expec.mean())

df3 = pd.concat([pd.Series([0,1,2,3,4]), hc_clst_PC1, hc_clst_PC2, hc_clst_PC3, hc_clst_PC4], axis=1)
df3.columns = ["ClusterID", "PC1_mean", "PC2_mean","PC3_mean","PC4_mean"]
df3

df4 = pd.concat([pd.Series([0,1,2,3,4]),hc_clst_child_mort,hc_clst_health,
                 hc_clst_total_fer,hc_clst_income,hc_clst_life_expec], axis=1)
df4.columns = ["ClusterID","child_mort_mean","health_mean","total_fer_mean","income_mean","life_expec_mean"]
df4

sns.barplot(x=df3.ClusterID, y=df3.PC1_mean)
sns.barplot(x=df3.ClusterID, y=df3.PC2_mean)
sns.barplot(x=df3.ClusterID, y=df3.PC3_mean)
sns.barplot(x=df3.ClusterID, y=df3.PC4_mean)
sns.barplot(x=df4.ClusterID, y=df4.child_mort_mean)
sns.barplot(x=df4.ClusterID, y=df4.health_mean)
sns.barplot(x=df4.ClusterID, y=df4.total_fer_mean)
sns.barplot(x=df4.ClusterID, y=df4.income_mean)
sns.barplot(x=df4.ClusterID, y=df4.life_expec_mean)

dfinal = df_hc[(df_hc.ClusterID==0)]

len(dfinal)
q = dfinal.income.quantile(.95)
dfinal.loc[(dfinal.income>=q)]

q = dfinal.child_mort.quantile(.05)
dfinal.loc[(dfinal.child_mort<q)]
q = dfinal.health.quantile(.95)
dfinal.loc[(dfinal.health>=q)]
dfinal.describe(percentiles=(.01,.05,.25,.50,.75,.95,.99))
country=pd.DataFrame(dfinal.country)

country.to_csv("Country.csv")
dfinal[dfinal.country=='Zambia']
