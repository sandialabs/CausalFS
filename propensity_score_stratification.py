import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize, KBinsDiscretizer
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings('ignore',category=UserWarning)
warnings.filterwarnings('ignore',category=ConvergenceWarning)

class PropensityScoreStratification(object):
	def __init__(self,num_classes=None,norm='l2',num_strata=5,
					clipping_threshold=10,weighted_average=True,
					verbose=False):
		'''
			num_classes
				- Number of classes for classification (>= 2)
				- If None, a regression model will be assumed
			norm
				- Type of normalization (L1, L2, Max)
				- If None, no normalization will be applied prior to quantifying causal effects
			num_strata
				- Number of strata to separate samples into when using propensity score stratification
			clipping_threshold
				- Minimum number of samples for any given stratum to be considered
					- (strata with < clipping_threshold samples will not be considered in causal quantification)
				- Must be >= 2

			Propensity score stratification determines causal impact (x->y) by first grouping samples by
				P(x|common_causes)
			where common_causes is the set of {C s.t C->x and C->y}. The causal impact is determined to be
				dy/dx ~y = f(x)
			the average linear regression for each strata. If weighted_average is True, the average for each
			strata will be weighted based on the R2 between x and ~y.
		'''
		assert num_classes is None or int(num_classes) >= 2, 'num_classes must be None (regression) or an integer >= 2'
		assert norm in [None,'l1','l2','max'], 'norm must be a string (l1, l2, max) or None (no normalization)'
		assert int(num_strata) >= 2, 'num_strata must be an integer >= 2 (default 5)'
		assert int(clipping_threshold) >= 2, 'clipping_threshold must be an integer >= 2 (default 10)'

		self.num_classes = num_classes
		self.norm = norm
		self.num_strata = num_strata
		self.clipping_threshold = clipping_threshold
		self.weighted_average = weighted_average
		self.verbose = verbose

	def print(self,message,verbose_only=True):
		'''
			verbose_only
				- Only print if self.verbose == True (default True)
		'''
		if (not verbose_only) or (verbose_only and self.verbose):
			tqdm.write(str(message))

	def estimate_causality(self, data_df, adjacency_matrix_df, y, X_features=None):
		''' 
			data_df 
				- A pandas Dataframe where rows are samples and columns are features.
				- Contains both X and y, where "y" is the quantity/classification to be predicted.
			adjacency_matrix_df
				- A pandas Dataframe with indices/columns == data_df.columns.
				- Individual values are 0/1 depending on potential causal relationships s.t. index->column
			y
				- The variable to be predicted
			X_features (optional)
				- If None, estimate causal effects of all X_features on y.
				- If type(X_features) is list, estimate causal effects of all X_features on y
				- If type(X_features) is string, estimate causal effect of a single X_feature on y,
					where X_feature in data_df.columns
		'''
		assert set(data_df.columns) == set(adjacency_matrix_df.columns) and set(data_df.columns) == set(adjacency_matrix_df.index), '\
				adjacency_matrix_df must be a Dataframe with indices/columns that reflect the data_df columns'
		assert X_features is None or type(X_features) is list or type(X_features) is str, '\
				X_features must be None, a list of X_features to evaluate, or a single X_feature (str)'
		assert y in data_df.columns, 'Feature %s not in data_df.columns' % y

		if X_features is None:
			X_features = [col for col in data_df.columns if col != y]
		elif type(X_features) is str:
			assert X_features in data_df.columns, 'Feature %s is not in data_df.columns' % X_features
			assert X_features != y, 'Cannot find causal effect of feature %s on itself' % y
			X_features = [X_features]
		elif type(X_features) is list:
			assert all([feat in data_df.columns for feat in X_features]), 'X_features must all be in data_df.columns'
			assert y not in X_features, 'Cannot find causal effect of feature %s on itself' % y


		# Normalize all_X_features to compare features of different scales
		self.print('Normalizing samples in data_df to compare features of different scale.')

		all_X_features = [col for col in data_df.columns if col != y]
		if self.norm is not None:
			data_df[all_X_features].iloc[:,:] = normalize(data_df[all_X_features].values,axis=0,norm=self.norm)


		# Discretize each of all_X_features using kmeans
		#	- Necessary for stratification via logistic regression
		self.print('Discretizing each feature for stratification via logistic regression')

		bin_df = data_df.copy()
		for feat in all_X_features:
			n_bins = min(self.num_strata,len(set(data_df[feat])))
			le = KBinsDiscretizer(n_bins=n_bins,encode='ordinal',strategy='kmeans')
			bin_df[feat] = le.fit_transform(data_df[feat].values.reshape(-1,1))


		# Iterate through X_features and estimate effect of (X_feature->y)
		self.print('Iterating through X_features to estimate effect (X_feature->y)')

		causal_df = pd.DataFrame(0,index=all_X_features,columns=[y])
		tqdm_features = tqdm(all_X_features)
		for feat in tqdm_features if self.verbose else all_X_features:
			if self.verbose:
				tqdm_features.set_description(feat)

			# Identify common causes (C such that C->feat and C->y)
			common_causes = adjacency_matrix_df.loc[(adjacency_matrix_df[[feat,y]]!=0).all(axis=1)].index
			if len(common_causes) == 0:
				causal_df.loc[feat,y] = np.nan
				self.print('No common causes between %s and %s, therefore cannot do propensity score stratification' % (feat,y))
				continue

			# Stratify samples
			strata_df = pd.DataFrame(index=data_df.index,columns=['treatment','propensity_score','strata','effect','r2'])
			strata_df['treatment'] = data_df[feat]

			propensity_score_model = LogisticRegression(solver='lbfgs',multi_class='auto',max_iter=500)
			propensity_score_model.fit(bin_df[common_causes], bin_df[feat])
			propensity_scores = propensity_score_model.predict_proba(bin_df[common_causes])

			for i in range(len(strata_df)):
				treatment = bin_df.loc[strata_df.index[i],feat]
				strata_df.loc[strata_df.index[i],'propensity_score'] = propensity_scores[i,int(treatment-1)]
			kmeans = KMeans(n_clusters=self.num_strata).fit(propensity_scores)
			strata_df['strata'] = kmeans.predict(propensity_scores)

			# Estimate dy/dx ~y = f(x) for each strata if there are sufficient # of samples
			for strata in sorted(set(strata_df['strata'])):
				tmp_df = strata_df.loc[strata_df['strata']==strata]
				samples = tmp_df.index

				if len(tmp_df) < self.clipping_threshold or len(set(tmp_df['treatment'])) < 2:
					strata_df.loc[samples,'effect'] = np.nan
					continue
				model = LinearRegression()
				tmpX = data_df.loc[samples,feat].values.reshape(-1,1)
				tmpy = data_df.loc[samples,y]
				model.fit(tmpX,tmpy)
				strata_df.loc[samples,'effect'] = model.coef_[0]
				strata_df.loc[samples,'r2'] = r2_score(tmpy,model.predict(tmpX))

			# Record average effect in causal_df
			strata_df = strata_df.dropna()
			if len(strata_df)==0 or len(set(strata_df['treatment']))==1:
				self.print('No valid strata for %s->%s' % (feat,y))
				causal_df.loc[feat,y] = np.nan
			else:
				if self.weighted_average:
					# Remove invalid R2
					strata_df.loc[strata_df['r2']<0,'r2'] = 0
					strata_df.loc[strata_df['r2']>1,'r2'] = 0
					wavg = (strata_df['effect'] * strata_df['r2']).sum() / strata_df['r2'].sum()
					causal_df.loc[feat,y] = wavg
					causal_df.loc[feat,'%s_R2' % y] = strata_df['r2'].mean()
					causal_df.loc[feat,'%s_NumValidSamples' % y] = len(strata_df)
				else:
					causal_df.loc[feat,y] = strata_df['effect'].mean()

			return causal_df
