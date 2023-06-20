import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import random
import csv
import matplotlib.pyplot as plt
import seaborn as sns
from lmfit import Parameters, minimize, report_fit
import math
M = 100
N = 100
locations = []
while len(locations) <= 99:
    s = set()
    while(len(s) <= 52):
        s.add((random.randint(0,100),random.randint(0,100))) 
    locations.append(list(s))
anchors = []
ranges = []
for lines in locations:
    line = [lines[0],lines[1],lines[2]]
    anchors.append(tuple(line))
    for i in range(3,len(lines)):
        r1 = np.linalg.norm(np.array(lines[i])-np.array(lines[0]))
        r2 = np.linalg.norm(np.array(lines[i])-np.array(lines[1]))
        r3 = np.linalg.norm(np.array(lines[i])-np.array(lines[2]))
        l = tuple((r1,r2,r3))
        ranges.append(l)
pure_ranges = []
true_locations = []
for i  in range(len(anchors)):
    l = []
    s = []
    l.append(anchors[i])
    s.append(anchors[i])
    start = 50*i
    end = 50*i + 50
    for j in range(start,end):
        l.append(ranges[j])
    s.append(locations[i][3:])
    pure_ranges.append(l)
    true_locations.append(s)
with open('true_locations.csv', 'w') as file:
    writer = csv.writer(file)
    writer.writerows(true_locations)
with open('pure_ranges.csv', 'w') as file:
    writer = csv.writer(file)
    writer.writerows(pure_ranges)
s = [0.5,1,2]
noisy_range = []
for i in s:
    string = "noisy_ranges_"+str(i)+".csv"
    noisy_ranges = []
    for j in range(100):
        l = []
        l.append(anchors[j])
        for k in range(50):
            original = np.array(pure_ranges[j][k+1])
            noise = np.random.normal(i, 0.1, original.shape)
            new_signal = original + noise
            tu = tuple(new_signal)
            l.append(tu)
        noisy_ranges.append(l)
    noisy_range.append(noisy_ranges)
    with open(string, 'w') as file:
        writer = csv.writer(file)
        writer.writerows(noisy_ranges)
A = random.randint(1,100)
B = random.randint(1,50)
anchor_locs = np.array(anchors[A-1])
true_loc = np.array(locations[A-1][B+2])
pure_range = np.array(pure_ranges[A-1][B])
noisy_range_0 = np.array(noisy_range[0][A-1][B])
noisy_range_1 = np.array(noisy_range[1][A-1][B])
noisy_range_2 = np.array(noisy_range[2][A-1][B])
print(anchor_locs,true_loc,pure_range,noisy_range_0,noisy_range_1,noisy_range_2)
range_for_map = [pure_range,noisy_range_0,noisy_range_1,noisy_range_2]
heatmaps = []
for r in range(len(range_for_map)):
    heatmap =[[] for i in range(100)]
    for j in range(100):
        for i in range(100):
            cost = 0
            for k in range(len(anchor_locs)):
                c = ((np.linalg.norm([i,j]-(anchor_locs[k])))-range_for_map[r][k])**2
                cost += c
            cost = np.sqrt(cost/3)
            heatmap[j].append(cost)
    heatmaps.append(heatmap)
names = ['pure_ranges','noisy_ranges_0.5','noisy_ranges_1','noisy_ranges_2']
for i in range(4):
    heatmaps[i] = np.array(heatmaps[i])
    min_of_map,max_of_map = heatmaps[i].min(),heatmaps[i].max()
    heatmaps[i] = (heatmaps[i] - min_of_map)/(max_of_map - min_of_map)
    print(np.var(heatmaps[i]))
    plt.title(names[i])
    plt.imshow(heatmaps[i],cmap = 'YlGnBu')
    plt.ylim(0, 100)
    plt.colorbar()
    plt.show()
'''
Using lmfit
'''
def residual(pars,x = None,data = None):
    x = pars['x']
    y = pars['y']
    x1 = pars['x1']
    x2 = pars['x2']
    x3 = pars['x3']
    y1 = pars['y1']
    y2 = pars['y2']
    y3 = pars['y3']
    r1 = pars['r1']
    r2 = pars['r2']
    r3 = pars['r3']

    cost = 0
    c1 = (np.sqrt((x-x1)**2 + (y-y1)**2) - r1)
    c2 = (np.sqrt((x-x2)**2 + (y-y2)**2) - r2)
    c3 = (np.sqrt((x-x3)**2 + (y-y3)**2) - r3)

    cost = np.sqrt((c1**2+c2**2+c3**2)//3)

    arr = np.zeros(3)
    arr[0] = c1
    arr[1] = c2
    arr[2] = c3
    return arr
arrays = [[] for i in range(4)]
for A in range(1,101):
    for B in range(1,51):
        anchor_locs = np.array(anchors[A-1])
        true_loc = np.array(locations[A-1][B+2])
        pure_range = np.array(pure_ranges[A-1][B])
        noisy_range_0 = np.array(noisy_range[0][A-1][B])
        noisy_range_1 = np.array(noisy_range[1][A-1][B])
        noisy_range_2 = np.array(noisy_range[2][A-1][B])
        range_for_map = [pure_range,noisy_range_0,noisy_range_1,noisy_range_2]
        params = Parameters()
        params.add('x1',value = anchor_locs[0][0], vary=False)
        params.add('y1',value = anchor_locs[0][1], vary=False)
        params.add('x2',value = anchor_locs[1][0], vary=False)
        params.add('y2',value = anchor_locs[1][1], vary=False)
        params.add('x3',value = anchor_locs[2][0], vary=False)
        params.add('y3',value = anchor_locs[2][1], vary=False)
        for r in range(len(range_for_map)):
            params.add('r1',value = range_for_map[r][0], vary=False)
            params.add('r2',value = range_for_map[r][1], vary=False)
            params.add('r3',value = range_for_map[r][2], vary=False)

            params.add('x',value = true_loc[0],max = 99,min = 0)
            params.add('y',value = true_loc[1],max = 99,min = 0)

            out = minimize(residual, params, args= None, kws=None)
            fit = residual(out.params, None)

            arrays[r].append(tuple((round(out.params['x'].value),round(out.params['y'].value))))
            

pure_locs = []
with open('pure_locs.csv', 'w') as file:
    for i  in range(len(anchors)):
        l = []
        l.append(anchors[i])
        start = 50*i
        end = 50*i + 50
        for j in range(start,end):
            l.append(arrays[0][j])
        pure_locs.append(l)
    writer = csv.writer(file)
    writer.writerows(pure_locs)
noisy_locs_05 = []
with open('noisy_locs_05.csv', 'w') as file:
    for i  in range(len(anchors)):
        l = []
        l.append(anchors[i])
        start = 50*i
        end = 50*i + 50
        for j in range(start,end):
            l.append(arrays[1][j])
        noisy_locs_05.append(l)
    writer = csv.writer(file)
    writer.writerows(noisy_locs_05)
noisy_locs_1 = []
with open('noisy_locs_1.csv', 'w') as file:
    for i  in range(len(anchors)):
        l = []
        l.append(anchors[i])
        start = 50*i
        end = 50*i + 50
        for j in range(start,end):
            l.append(arrays[2][j])
        noisy_locs_1.append(l)
    writer = csv.writer(file)
    writer.writerows(noisy_locs_1)
noisy_locs_2 = []
with open('noisy_locs_2.csv', 'w') as file:
    for i  in range(len(anchors)):
        l = []
        l.append(anchors[i])
        start = 50*i
        end = 50*i + 50
        for j in range(start,end):
            l.append(arrays[3][j])
        noisy_locs_2.append(l)
    writer = csv.writer(file)
    writer.writerows(noisy_locs_2)
errors = [[] for i in range(4)]
locs = [pure_locs,noisy_locs_05,noisy_locs_1,noisy_locs_2]
for loc in range(4):
    for i in range(len(locs[loc])):
        for j in range(len(locs[loc][i])-1):
            error = np.linalg.norm(np.array(true_locations[i][1][j])-np.array(locs[loc][i][j+1]))
            errors[loc].append(error)
cdf_errors = [np.cumsum(errors[0]),np.cumsum(errors[1]),np.cumsum(errors[2]),np.cumsum(errors[3])]
x = [i for i in range(5000)]
for i in range(4):
    plt.plot(x,cdf_errors[i])
    # plt.show()
names = ["pure_locs","noisy_locs_05","noisy_locs_1","noisy_locs_2"]
plt.legend(names)
plt.show()
medians = []
percentile_75 = []
percentile_95 = []
for i in range(4):
    medians.append(np.median(np.array(errors[i])))
    print("Median for errors of "+names[i]+" is "+str(medians[i]))
    percentile_75.append(np.percentile(np.array(errors[i]),75))
    print("75 percentile error of "+names[i]+" is "+str(percentile_75[i]))
    percentile_95.append(np.percentile(np.array(errors[i]),95))
    print("95 percentile error of "+names[i]+" is "+str(percentile_95[i]))

median_localization_errors = [[] for i in range(4)] 
for i in range(4):
    for j in range(100):
        median_localization_error = np.median(errors[i][j])
        median_localization_errors[i].append(median_localization_error)

for i in range(4):
    plt.hist(median_localization_errors[i])
    plt.xlim(xmin=np.array(median_localization_errors[0]).min(), xmax = np.array(median_localization_errors[3]).max())
    plt.show() 