import pandas as pd
a =[12,3,4]
b = [22,33,44]

a = pd.DataFrame(a)
b = pd.DataFrame(b)

for d1,d2 in zip(a.iterrows(), b.iterrows()):
    print('\n\n\n')
    print(d1[0],d2[1])
    d2[1][0] +=10
for i,d2 in b.iterrows():
    print(d2)