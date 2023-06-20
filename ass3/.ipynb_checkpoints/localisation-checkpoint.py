import numpy as np
import scipy.io.wavfile
import scipy.signal
import matplotlib.pyplot as plt
from scipy import signal
import math
import statistics
from tabulate import tabulate
import random
import seaborn as sns
from lmfit import minimize, Parameters
grid=[[0] * 100] * 100
tuples=[]
anchor_set=set()
for i in range(100):
	for j in range(100):
		tuples.append((i,j))
while len(anchor_set)<100:
	random.shuffle(tuples)
	t1=tuples.pop()
	random.shuffle(tuples)
	t2=tuples.pop()
	random.shuffle(tuples)
	t3=tuples.pop()
	anchor_set.add((t1,t2,t3))
	tuples.append(t1)
	tuples.append(t2)
	tuples.append(t3)
anchors=np.array(list(anchor_set))
alln=[]
ra1=[]
ra2=[]
ra3=[]
ra4=[]
for i in range(100):
	temp=set()
	while len(temp)<50:
		temp.add(random.choice(tuples))
	nodes=np.array(list(temp))
	alln.append(nodes)
allnar=np.array(alln)
f1 = open("true_locations.csv","w")
f2 = open("pure_ranges.csv","w")
f3=open("noisy_ranges_05.csv","w")
f4=open("noisy_ranges_1.csv","w")
f5=open("noisy_ranges_2.csv","w")
for i in range(100):
	s="["+"("+str(anchors[i][0][0])+","+str(anchors[i][0][1])+")"+","+"("+str(anchors[i][1][0])+","+str(anchors[i][1][1])+")"+","+"("+str(anchors[i][2][0])+","+str(anchors[i][2][1])+")"+"]"+str(i+1)
	s1=s
	s2=s
	s3=s
	s4=s
	p_ranges=[]
	n_ranges05=[]
	n_ranges1=[]
	n_ranges2=[]
	for j in range(50):
		s=s+","+"("+str(allnar[i][j][0])+","+str(allnar[i][j][1])+")"
		r1=np.linalg.norm(anchors[i][0] - allnar[i][j])
		r2=np.linalg.norm(anchors[i][1] - allnar[i][j])
		r3=np.linalg.norm(anchors[i][2] - allnar[i][j])
		p_ranges.append(np.array((r1,r2,r3)))
		r105=r1+np.random.normal(0.5, 0.1)
		r205=r2+np.random.normal(0.5, 0.1)
		r305=r3+np.random.normal(0.5, 0.1)
		r11=r1+np.random.normal(1, 0.1)
		r21=r2+np.random.normal(1, 0.1)
		r31=r3+np.random.normal(1, 0.1)
		r12=r1+np.random.normal(2, 0.1)
		r22=r2+np.random.normal(2, 0.1)
		r32=r3+np.random.normal(2, 0.1)
		n_ranges05.append(np.array((r105,r205,r305)))
		n_ranges1.append(np.array((r11,r21,r31)))
		n_ranges2.append(np.array((r12,r22,r32)))
		s1=s1+","+"("+str(r1)+","+str(r2)+","+str(r3)+")"+str(j+1)
		s2=s2+","+"("+str(r105)+","+str(r205)+","+str(r305)+")"+str(j+1)
		s3=s3+","+"("+str(r11)+","+str(r21)+","+str(r31)+")"+str(j+1)
		s4=s4+","+"("+str(r12)+","+str(r22)+","+str(r32)+")"+str(j+1)
	f1.write(s)
	f2.write(s1)
	f3.write(s2)
	f4.write(s3)
	f5.write(s4)
	ra1.append(p_ranges)
	ra2.append(n_ranges05)
	ra3.append(n_ranges1)
	ra4.append(n_ranges2)

f1.close()
f2.close()
f3.close()
f4.close()
f5.close()
a=random.randrange(0, 99)
b=random.randrange(0, 49)
print("Random true location that is selected:",end=' ')
print(allnar[a][b])
h1=np.array([np.array([0] *100)] *100)
h2=np.array([np.array([0] *100)] *100)
h3=np.array([np.array([0] *100)] *100)
h4=np.array([np.array([0] *100)] *100)
for i in range(100):
	for j in range(100):
		cost=((np.linalg.norm(np.array([i,j])-anchors[a][0])-ra1[a][b][0])**2+(np.linalg.norm(np.array([i,j])-anchors[a][1])-ra1[a][b][1])**2+(np.linalg.norm(np.array([i,j])-anchors[a][2])-ra1[a][b][2])**2)/3
		cost=math.sqrt(cost)
		h1[i][j]=cost

		cost=((np.linalg.norm(np.array([i,j])- anchors[a][0])-ra2[a][b][0])**2+(np.linalg.norm(np.array([i,j])- anchors[a][1])-ra2[a][b][1])**2+(np.linalg.norm(np.array([i,j])- anchors[a][2])-ra2[a][b][2])**2)/3
		cost=math.sqrt(cost)
		h2[i][j]=cost

		cost=((np.linalg.norm(np.array([i,j])- anchors[a][0])-ra3[a][b][0])**2+(np.linalg.norm(np.array([i,j])- anchors[a][1])-ra3[a][b][1])**2+(np.linalg.norm(np.array([i,j])- anchors[a][2])-ra3[a][b][2])**2)/3
		cost=math.sqrt(cost)
		h3[i][j]=cost

		cost=((np.linalg.norm(np.array([i,j])- anchors[a][0])-ra4[a][b][0])**2+(np.linalg.norm(np.array([i,j])- anchors[a][1])-ra4[a][b][1])**2+(np.linalg.norm(np.array([i,j])- anchors[a][2])-ra4[a][b][2])**2)/3
		cost=math.sqrt(cost)
		h4[i][j]=cost

hmin,hmax=h1.min(),h1.max()
h1 = (h1 - hmin)/(hmax - hmin)
ax = sns.heatmap(h1, cmap="YlGnBu")
ax.invert_yaxis()
plt.title('Heat map for pure ranges')
plt.show()

hmin,hmax=h2.min(),h2.max()
h2 = (h2 - hmin)/(hmax - hmin)
ax = sns.heatmap(h2, cmap="YlGnBu")
ax.invert_yaxis()
plt.title('Heat map for noisy ranges 0.5')
plt.show()

hmin,hmax=h3.min(),h3.max()
h3 = (h3 - hmin)/(hmax - hmin)
ax = sns.heatmap(h3, cmap="YlGnBu")
ax.invert_yaxis()
plt.title('Heat map for noisy ranges 1')
plt.show()

hmin,hmax=h4.min(),h4.max()
h4 = (h4 - hmin)/(hmax - hmin)
ax = sns.heatmap(h4, cmap="YlGnBu")
ax.invert_yaxis()
plt.title('Heat map for noisy ranges 2')
plt.show()

pointsl=[]

for i in range(100):
	for j in range(100):
		pointsl.append((i,j))
f6=open("true_locs.csv","w")
f7=open("noisy_locs_05.csv","w")
f8=open("noisy_locs_1.csv","w")
f9=open("noisy_locs_2.csv","w")
def residual(params):
	a1x=params['a1x']
	a1y=params['a1y']
	a2x=params['a2x']
	a2y=params['a2y']
	a3x=params['a3x']
	a3y=params['a3y']
	ra1=params['ra1']
	ra2=params['ra2']
	ra3=params['ra3']
	x1=params['x']
	y1=params['y']
	x=[x1,y1]
	c1=(np.linalg.norm(x- np.array([a1x,a1y]))-ra1)
	c2=(np.linalg.norm(x- np.array([a2x,a2y]))-ra2)
	c3=(np.linalg.norm(x- np.array([a3x,a3y]))-ra3)
	return np.array([c1,c2,c3])
err1=[]
err2=[]
err3=[]
err4=[]
me1=[]
me2=[]
me3=[]
me4=[]
for i in range(100):
	s="["+"("+str(anchors[i][0][0])+","+str(anchors[i][0][1])+")"+","+"("+str(anchors[i][1][0])+","+str(anchors[i][1][1])+")"+","+"("+str(anchors[i][2][0])+","+str(anchors[i][2][1])+")"+"]"+str(i+1)
	s1=s
	s2=s
	s3=s
	s4=s
	e1=[]
	e2=[]
	e3=[]
	e4=[]
	for j in range(50):
		params = Parameters()
		params.add('a1x', value=anchors[i][0][0],vary=False)
		params.add('a1y', value=anchors[i][0][1],vary=False)
		params.add('a2x', value=anchors[i][1][0],vary=False)
		params.add('a2y', value=anchors[i][1][1],vary=False)
		params.add('a3x', value=anchors[i][2][0],vary=False)
		params.add('a3y', value=anchors[i][2][1],vary=False)
		params.add('ra1', value=ra1[i][j][0],vary=False)
		params.add('ra2', value=ra1[i][j][1],vary=False)
		params.add('ra3', value=ra1[i][j][2],vary=False)
		params.add('x',value = allnar[i][j][0],max = 99,min = 0)
		params.add('y',value = allnar[i][j][1],max = 99,min = 0)

		out = minimize(residual, params)
		fit = residual(out.params)
		s1=s1+","+"("+str(round(out.params['x'].value))+","+str(round(out.params['y'].value))+")"
		err1.append(np.linalg.norm(allnar[i][j]- np.array([round(out.params['x'].value),round(out.params['y'].value)])))
		e1.append(np.linalg.norm(allnar[i][j]- np.array([round(out.params['x'].value),round(out.params['y'].value)])))

		params.add('ra1', value=ra2[i][j][0],vary=False)
		params.add('ra2', value=ra2[i][j][1],vary=False)
		params.add('ra3', value=ra2[i][j][2],vary=False)

		out = minimize(residual, params)
		fit = residual(out.params)
		s2=s2+","+"("+str(round(out.params['x'].value))+","+str(round(out.params['y'].value))+")"
		err2.append(np.linalg.norm(allnar[i][j]- np.array([round(out.params['x'].value),round(out.params['y'].value)])))
		e2.append(np.linalg.norm(allnar[i][j]- np.array([round(out.params['x'].value),round(out.params['y'].value)])))

		params.add('ra1', value=ra3[i][j][0],vary=False)
		params.add('ra2', value=ra3[i][j][1],vary=False)
		params.add('ra3', value=ra3[i][j][2],vary=False)

		out = minimize(residual, params)
		fit = residual(out.params)
		s3=s3+","+"("+str(round(out.params['x'].value))+","+str(round(out.params['y'].value))+")"
		err3.append(np.linalg.norm(allnar[i][j]- np.array([round(out.params['x'].value),round(out.params['y'].value)])))
		e3.append(np.linalg.norm(allnar[i][j]- np.array([round(out.params['x'].value),round(out.params['y'].value)])))

		params.add('ra1', value=ra4[i][j][0],vary=False)
		params.add('ra2', value=ra4[i][j][1],vary=False)
		params.add('ra3', value=ra4[i][j][2],vary=False)

		out = minimize(residual, params)
		fit = residual(out.params)
		s4=s4+","+"("+str(round(out.params['x'].value))+","+str(round(out.params['y'].value))+")"
		err4.append(np.linalg.norm(allnar[i][j]- np.array([round(out.params['x'].value),round(out.params['y'].value)])))
		e4.append(np.linalg.norm(allnar[i][j]- np.array([round(out.params['x'].value),round(out.params['y'].value)])))
	f6.write(s1)
	f7.write(s2)
	f8.write(s3)
	f9.write(s4)
	me1.append(np.median(e1))
	me2.append(np.median(e2))
	me3.append(np.median(e3))
	me4.append(np.median(e4))


f6.close()
f7.close()
f8.close()
f9.close()
sns.ecdfplot(err1,label='Pure ranges')
sns.ecdfplot(err2,label='Noisy ranges 0.5')
sns.ecdfplot(err3,label='Noisy ranges 1')
sns.ecdfplot(err4,label='Noisy ranges 2')
plt.legend()
plt.title('CDF of Error')
plt.xlabel('Error')
plt.show()
print("For pure ranges:")
print("Median error:",end=' ')
print(np.median(err1))
# print(np.mean(err1))
print("75th percentile error:",end=' ')
print(np.percentile(np.array(err1),75))
print("95th percentile error:",end=' ')
print(np.percentile(np.array(err1),95))
print("For noisy ranges 0.5:")
print("Median error:",end=' ')
print(np.median(err2))
# print(np.mean(err2))
print("75th percentile error:",end=' ')
print(np.percentile(np.array(err2),75))
print("95th percentile error:",end=' ')
print(np.percentile(np.array(err2),95))
print("For noisy ranges 1:")
print("Median error:",end=' ')
print(np.median(err3))
# print(np.mean(err3))
print("75th percentile error:",end=' ')
print(np.percentile(np.array(err2),75))
print("95th percentile error:",end=' ')
print(np.percentile(np.array(err2),95))
print("For noisy ranges 2:")
print("Median error:",end=' ')
print(np.median(err4))
# print(np.mean(err4))
print("75th percentile error:",end=' ')
print(np.percentile(np.array(err4),75))
print("95th percentile error:",end=' ')
print(np.percentile(np.array(err4),95))
plt.hist(me1)
plt.title('Histogram of Median Error with each anchor set for pure ranges')
plt.xlabel('Median Error')
plt.xlim(0,4)
plt.show()
plt.hist(me2)
plt.title('Histogram of Median Error with each anchor set for noisy ranges 0.5')
plt.xlabel('Median Error')
plt.xlim(0,4)
plt.show()
plt.hist(me3)
plt.title('Histogram of Median Error with each anchor set for noisy ranges 1')
plt.xlabel('Median Error')
plt.xlim(0,4)
plt.show()
plt.hist(me4)
plt.title('Histogram of Median Error with each anchor set for noisy ranges 2')
plt.xlabel('Median Error')
plt.xlim(0,4)
plt.show()