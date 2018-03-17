from pandas import read_csv
from matplotlib import pyplot

data = read_csv('training-set.csv')
data.plot(kind='density', subplots=True, layout=(7,7), sharex=False, legend=False,fontsize=1)
pyplot.show()