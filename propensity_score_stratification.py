import pandas as pd
import numpy as np
import networkx as nx
from tqdm import tqdm
from sklearn.preprocessing import normalize, KBinsDiscretizer

class PropensityScoreStratification(object):
	def __init__(self,num_classes=None,normalize='l2',num_strata=5,clipping_threshold=10,verbose=False):
		'''
			num_classes
				- Number of classes for classification (>= 2)
				- If None, a regression model will be assumed
			normalize
				- Type of normalization (L1, L2, Max)
				- If None, no normalization will be applied prior to quantifying causal effects
			num_strata
				- Number of strata to separate samples into when using propensity score stratification
			clipping_threshold
				- Minimum number of samples for any given stratum to be considered
					- (strata with < clipping_threshold samples will not be considered in causal quantification)
				- Must be >= 2
		'''
		assert num_classes is None or int(num_classes) >= 2, 'num_classes must be None (regression) or an integer >= 2'
		assert normalize in [None,'l1','l2','max'], 'normalize must be a string (l1, l2, max) or None (no normalization)'
		assert int(num_strata) >= 2, 'num_strata must be an integer >= 2 (default 5)'
		assert int(clipping_threshold) >= 2, 'clipping_threshold must be an integer >= 2 (default 10)'

		self.num_classes = num_classes
		self.normalize = normalize
		self.num_strata = num_strata
		self.clipping_threshold = clipping_threshold
		self.verbose = verbose

	def print(self,message,verbose_only=True):
		'''
			verbose_only
				- Only print if self.verbose == True (default True)
		'''
		if (not verbose_only) or (verbose_only and self.verbose):
			print(message)

	def estimate_causality(self, data_df, adjacency_matrix_df, y, X_features=None):
		''' 
			data_df 
				- A pandas Dataframe where rows are samples and columns are features.
				- Contains both X and y, where "y" is the quantity/classification to be predicted.
			adjacency_matrix_df
				- A pandas Dataframe with indices/columns == data_df.columns.
				- Individual values are 0/1 depending on potential causal relationships s.t. index -> column
			y
				- The variable to be predicted
			X_features (optional)
				- If None, estimate causal effects of all X_features on y.
				- If type(X_features) is list, estimate causal effects of all X_features on y
				- If type(X_features) is string, estimate causal effect of a single X_feature on y,
					where X_feature in data_df.columns
		'''
		assert data_df.columns == adjacency_matrix_df.columns and data_df.columns == adjacency_matrix_df.index, '\
				adjacency_matrix_df must be a Dataframe with indices/columns that reflect the data_df columns'
		assert X_features is None or type(X_features) is list or type(X_features) is str, '\
				X_features must be None, a list of X_features to evaluate, or a single X_feature (str)'
		if X_features is None:
			X_features = [col for col in data_df.columns if col != y]
		elif type(X_features) is str:
			X_features = [X_features]

