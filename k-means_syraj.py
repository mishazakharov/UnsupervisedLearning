import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#
# Загружает данные
def load_dataset(name):
	return np.loadtxt(name)

def euclidian(a,b):
	return np.linalg.norm(a-b)

def kmeans(k,epsilon=0.2,distance='euclidian'):
	''' Тело алгоритма '''
	# Список с прошлыми центроидами
	history_centroids = []
	# Устанавливаем тип измерения расстояний
	if distance == 'euclidian':
		dist_method = 'euclidian'
	# Dataset
	dataset = load_dataset('durudataset.txt')
	# количество сэмплов и признаков
	num_instances,num_features = dataset.shape
	# определяем  к-центроидов
	prototypes = dataset[np.random.randint(0,num_instances-1,size=k)]
	# заносим в наш список хранения
	history_centroids.append(prototypes)
	# чтобы следить за центроидами на каждой итерации
	prototypes_old = np.zeros(prototypes.shape)
	# хранить кластеры
	belongs_to = np.zeros((num_instances,1))
	norm = euclidian(prototypes,prototypes_old)
	iteration = 0

	while norm > epsilon:
		iteration += 1
		norm = euclidian(prototypes,prototypes_old)
		# для каждого сэмпла в данных
		for index_instance,instance in enumerate(dataset):
			# определяем вектор расстояния размером К
			dist_vect = np.zeros((k,1))
			# для каждого цетроида
			for index_prototype,prototype in enumerate(prototypes):
				# вычисляем расстояние между x и центроидом
				dist_vect[index_prototype] = euclidian(prototype,instance)
			# находим минимальное расстояние, включаем в кластер
			belongs_to[index_instance,0] = np.argmin(dist_vect)+1

		tmp_prototypes = np.zeros((k,num_features))

		# для каждого кластера
		for index in range(len(prototypes)):
			# берем все точки, принадлежащие кластеру
			instances_close = ([i for i in range(len(belongs_to)) 
										if belongs_to[i] == index])
			# находим среднее
			prototype = np.mean(dataset[instances_close],axis=0)
			# добавляем новый центроид в список временных центроидов
			tmp_prototypes[index,:] = prototype

		prototypes = tmp_prototypes
		history_centroids.append(tmp_prototypes)

	return belongs_to

c = kmeans(3)
print(c)