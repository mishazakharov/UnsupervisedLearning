import apyori as apriori
import numpy as np
import pandas as pd

'''
# Создадим маленький датасет
dataset = np.array([[1,1,1],[1,1,1],[1,1,1],[1,1,1]])
dataset = pd.DataFrame(dataset)
dataset.columns = ['Apple','Watermelon','Orange']
'''

# загрузим датасет
dataset = pd.read_csv('store_data.csv',header=None)

# дополенние путсых мест
dataset.fillna(method = 'ffill',axis = 1, inplace = True)
dataset.head()

# Создание матрицы из датасета
records = []  
for i in range(0,101):  
    records.append([str(dataset.values[i,j]) for j in range(0,20)])

# Применение алгоримта через библиотеку
result = list(apriori.apriori(records,min_support=0.003,
						min_confidence=0.2,min_lift=4,min_length=2))

print(result[0])

'''
# Вывод
for item in result:

    # first index of the inner list
    # Contains base item and add item
    pair = item[0] 
    items = [x for x in pair]
    print("Rule: " + items[0] + " -> " + items[1])

    #second index of the inner list
    print("Support: " + str(item[1]))

    #third index of the list located at 0th
    #of the third index of the inner list

    print("Confidence: " + str(item[2][0][2]))
    print("Lift: " + str(item[2][0][3]))
    print("=====================================")
'''