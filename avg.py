import statistics
acc = []
f1 = []
with open("result_snow.txt", "r") as f:
    for line in f.readlines():
        acc.append(float(line.strip()))
        #f1.append(float(line.split()[1].strip()))
print("result: ",statistics.mean(acc), statistics.stdev(acc))
#print(statistics.mean(f1), statistics.stdev(f1))
'''
with open("result_mix.txt", "r") as f:
    for line in f.readlines():
        acc.append(float(line.strip()))
        #f1.append(float(line.split()[1].strip()))
print("mix: ",statistics.mean(acc), statistics.stdev(acc))
#print(statistics.mean(f1), statistics.stdev(f1))
with open("result_aug.txt", "r") as f:
    for line in f.readlines():
        acc.append(float(line.strip()))
        #f1.append(float(line.split()[1].strip()))
print("aug: ",statistics.mean(acc), statistics.stdev(acc))
#print(statistics.mean(f1), statistics.stdev(f1))
'''
