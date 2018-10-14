import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt 
from scipy import stats
from scipy.stats import skew, norm

class FeatureInsight(object):
	"""The class is used to draw plots to get insight on data"""

	def __init__(self, feature, target):
		if type(feature) is not pd.DataFrame:
			raise ValueError("Expected argument of type pd.DataFrame")
		if type(target) is not pd.Series:
			raise ValueError("Expected argument of type pd.Series")

		self.feature = feature
		self.target = target

	def target_analysis(self):
		"""get distribution curve with mu and sigma values,
		get skewness and kutosis and target decription and get qq-plot"""

		print(self.target.describe())

		sns.distplot(self.target, fit = norm)
		plt.xlabel("target")
		plt.ylabel("frequency")
		plt.title("distribution plot")

		mu, sigma = norm.fit(self.target.values)
		print("mu={:.4f}\nsigma={:.4f}".format(mu, sigma))

		skewness = self.target.skew()
		kurtosis = self.target.kurt()
		print("skewness = {:.4f}\nkurtosis = {:.4f}".format(skewness, kurtosis))

		fig = plt.figure()
		res = stats.probplot(self.target, plot = plt)
		plt.show()

	def miss_my_data(self, all_data):
		"""to get insight on missing values"""

		if all_data is not pd.DataFrame:
			raise ValueError("Expected argument of type pd.DataFrame")

        miss_data = (all_data.isnull().sum()/len(all_data))*100

		miss_data = miss_data.drop(miss_data[miss_data == 0].index).sort_values(ascending = False)

		miss_data = pd.DataFrame({
		    "MissingVal" : miss_data
		})

		return miss_data.head(10)

	def heat_my_maps(self):
		"""To gather insight on multicolinear data"""

		corrmat = self.feature.corr()
		f, ax = plt.sublots((12, 9))
		sns.heatmap(corrmat, vmax = 0.9, square = True)