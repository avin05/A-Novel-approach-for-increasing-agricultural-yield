import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)
import csv
from pandas.plotting import scatter_matrix

def process(path):
	np.random.seed(0)
	data = pd.read_csv(path)
	print(data.head())
	names=list(data.columns)
	print(data.iloc[:,0:1])
	ss=[]
	ss1=[]
	for line in open(path):
		print(line)
		csv_row = line.split(",") 
		print(csv_row[0],csv_row[1],csv_row[2],csv_row[3])
		a=int(csv_row[0])
		b=int(csv_row[1])
		c=int(csv_row[2])
		d=int(csv_row[3])
		e=int(csv_row[4])
		f="NO"
		g=0
		if (a >= 20 and a <= 25 and b >= 55 and b <= 58 and c >= 85 and c <= 100 and d >= 1100 and d <= 2200 and e>=6.0 and e<= 7.0):
			f = "Wheat"
			g=1
		elif (a >= 23 and a <= 25 and b >= 65 and b <= 75 and c >= 70 and c <= 90 and d >= 200 and d <= 400 and e>=5.0 and e<= 7.0):
			f = "Oats"
			g=2
			
		elif (a >= 20 and a <= 40 and b >= 55 and b <= 70 and c >= 60 and c <= 80 and d >= 600 and d <= 750 and e>=6.0 and e<= 8.0):
			f = "Gram"
			g=3
			
		elif (a >= 12 and a <= 25 and b >= 52 and b <= 67 and c >= 85 and c <= 95 and d >= 450 and d <= 850 and e>=6.0 and e<= 7.0):
			f = "Pea"
			g=4
			
		elif (a >= 27 and a <= 32 and b >= 68 and b <= 80 and c >= 70 and c <= 88 and d >= 350 and d <= 440 and e>=6.0 and e<= 7.0):
			f = "Millets"
			g=5
			
		elif (a >= 20 and a <= 30 and b >= 60 and b <= 80 and c >= 60 and c <= 80 and d >= 450 and d <= 700 and e>=6.0 and e<= 7.0):
			f = "Rice"
			g=6
			
		elif (a >= 26 and a <= 34 and b >= 55 and b <= 72 and c >= 81 and c <= 95 and d >= 460 and d <= 590 and e>=5.0 and e<= 7.0):
			f = "Bajra"
			g=7
			
		elif (a >= 18 and a <= 27 and b >= 50 and b <= 65 and c >= 75 and c <= 90 and d >= 860 and d <= 1000 and e>=6.0 and e<= 7.0):
			f = "Maize"
			g=8
			
		elif (a >= 15 and a <= 19 and b >= 65 and b <= 83 and c >= 75 and c <= 85 and d >= 1350 and d <= 2000 and e>=6.0 and e<= 8.0):
			f = "Potato"
			g=9
			
		elif (a >= 26 and a <= 32 and b >= 72 and b <= 82 and c >= 78 and c <= 85 and d >= 1020 and d <= 1500 and e>=5.0 and e<= 7.0):
			f = "Groundnut"
			g=10
			
		elif (a >= 24 and a <= 38 and b >= 60 and b <= 95 and c >= 65 and c <= 75 and d >= 950 and d <= 1300 and e>=4.0 and e<= 6.0):
			f = "Jute"
			g=11
			
		elif (a >= 22 and a <= 32 and b >= 84 and b <= 88 and c >= 77 and c <= 90 and d >= 1050 and d <= 2200 and e>=6.0 and e<= 9.0):
			f = "Sugarcane"
			g=12
			
		elif (a >= 20 and a <= 30 and b >= 89 and b <= 100 and c >= 80 and c <= 90 and d >= 1100 and d <= 1700 and e>=6.0 and e<=7.0):
			f = "Turmeric"
			g=13
			
		elif (a >= 0 and a <= 19):
			f = "No Crop"
			g=14
		elif (b >= 0 and b <= 50):
                        f = "No Crop"
                        g=14
                
		print(f)
		print(g)
		s=[a,b,c,d,e,f]
		s1=[a,b,c,d,e,g]
		ss.append(s)
		ss1.append(s1)
	print(ss)
	print(ss1)

	with open('results/data1.csv', 'w',newline='') as myFile:
		writer = csv.writer(myFile)
		writer.writerows(ss)
	
	with open('results/data2.csv', 'w', newline='') as myFile:
		writer = csv.writer(myFile)
		writer.writerows(ss1)
	
	data = pd.read_csv('data_labels.csv')
	print(data.head())

	names=list(data.columns)



	correlations = data.corr()
	# plot correlation matrix
	fig = plt.figure()
	fig.canvas.set_window_title('Correlation Matrix')
	ax = fig.add_subplot(111)
	cax = ax.matshow(correlations, vmin=-1, vmax=1)
	fig.colorbar(cax)
	ticks = np.arange(0,9,1)
	ax.set_xticks(ticks)
	ax.set_yticks(ticks)
	ax.set_xticklabels(names)
	ax.set_yticklabels(names)
	fig.savefig('results/Correlation Matrix.png')
	
	plt.show()
	plt.close()
    
 
	#scatterplot

        
	#scatter_matrix(data)
	#plt.savefig('results/scattermatrix.png')
	
	#plt.show()
	#plt.close()

	fig2 = plt.gcf()
	fig2.canvas.set_window_title('Data Histogram')
	ncols=3
	plt.clf()
	f = plt.figure(1)
	f.suptitle("Data Histograms", fontsize=12)
	vlist = list(data.columns)
	nrows = len(vlist) // ncols
	if len(vlist) % ncols > 0:
		nrows += 1
	for i, var in enumerate(vlist):
		plt.subplot(nrows, ncols, i+1)
		plt.hist(data[var].values, bins=15)
		plt.title(var, fontsize=10)
		plt.tick_params(labelbottom='off', labelleft='off')
	plt.tight_layout()
	plt.subplots_adjust(top=0.88)
	plt.savefig('results/DataHistograms.png')
	plt.show()
	plt.close()
