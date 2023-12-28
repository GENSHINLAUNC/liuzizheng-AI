import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation


# 将文件中的城市坐标存到城市列表里
def load_data(city_axis):

    cities = []
    with open(city_axis, 'r') as f:
        lines = f.readlines()
        for line in lines:
            x_str, y_str = line.split()[1:] #第一个元素是序号，不需要
            x, y = int(x_str), int(y_str)
            cities.append((x, y))
    return cities

#计算城市之间的距离，存入距离矩阵里
def get_cities_distance(cities):

    dist_matrix = np.zeros((len(cities), len(cities)))
    num_cities = len(cities)
    #在纸上画一画可以写出这个循环，不会有重复计算，每个也算到
    for i in range(num_cities - 1):
        for j in range(i + 1, num_cities):
            x1,y1=cities[i]
            x2,y2=cities[j]
            dist=math.sqrt(math.pow(x1-x2,2)+math.pow(y1-y2,2))
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist
    return dist_matrix

#获取每条路径的评估值
def get_all_routes_fitness(routes, dist_matrix):

    fitness = np.zeros(len(routes))
    for i in range(len(routes)):
        dist_sum=0
        oneroute=routes[i]
        for j in range (0,len(oneroute)-1) :
            dist_sum+=dist_matrix[oneroute[j],oneroute[j+1]]
        #旅行商最后要回到起点
        dist_sum+=dist_matrix[oneroute[len(oneroute)-1],oneroute[0]]
        #求的是最短路径，fitness与距离成反比
        fitness[i] = 1/dist_sum
    return fitness

#路径数量(就是种群里个体的数量),城市数(就是染色体上基因的数量)
def init_route(route_num, cities_num):
   
    routes = np.zeros((route_num, cities_num)).astype(int)
    #不能重复选，每个城市都要走到且只走一次
    for i in range(route_num):
        routes[i] = np.random.choice(range(cities_num), size=cities_num, replace=False)
    return routes


def selection(routes, fitness_values):
  
    #创建一个与routes形状完全相同的全0整数数组
    selected_routes = np.zeros(routes.shape).astype(int)
    probability = fitness_values / np.sum(fitness_values)
    #shape第0个是行，第1个是列，如果是三维数组那么第2个就是高,行就是个数也就是种群个体数量
    for i in range(routes.shape[0]):
        #这里必须replace=TRUE，即可以重复选，这样才能体现自然选择
        choice = np.random.choice(range(routes.shape[0]), p=probability)
        selected_routes[i] = routes[choice]
    return selected_routes

#changeplace后面的两者正常互换，changeplace前面的需要和已经互换后面的互补加起来是一条完整的路径
def crossover(routes, cities_num):

    for i in range(0, len(routes), 2):
        r1_new, r2_new = np.zeros(cities_num), np.zeros(cities_num)
        changeplace = np.random.randint(0, cities_num)
        cross_len = cities_num - changeplace
        r1, r2 = routes[i], routes[i + 1]
        r1_cross, r2_cross = r2[changeplace:], r1[changeplace:]
        #选在r1的但是不在r1_cross内的
        r1_not_cross = r1[np.in1d(r1, r1_cross, invert=True)]
        r2_not_cross = r2[np.in1d(r2, r2_cross, invert=True)]
        #一拼，non_cross在前，cross在后
        r1_new[:cross_len], r2_new[:cross_len] = r1_cross, r2_cross
        r1_new[cross_len:], r2_new[cross_len:] = r1_not_cross, r2_not_cross
        routes[i], routes[i + 1] = r1_new, r2_new
    return routes


def mutation(routes, n_cities):
  
    mutation_rate = 0.1
    p_rand = np.random.rand(len(routes))
    for i in range(len(routes)):
        if p_rand[i] <= mutation_rate:
            mut_position = np.random.choice(range(n_cities), size=2, replace=False)
            l, r = mut_position[0], mut_position[1]
            routes[i, l], routes[i, r] = routes[i, r], routes[i, l]
    return routes


route_num = 100  # 种群中个体
generations = 100000  # 迭代次数

cities = load_data('cities.txt')  
dist_matrix = get_cities_distance(cities)  # 计算城市距离矩阵
routes = init_route(route_num, dist_matrix.shape[0])  # 初始化所有路线，第一代
fitness = get_all_routes_fitness(routes, dist_matrix)  # 计算所有初始路线的适应度
best_index = fitness.argmax()
best_route, best_fitness = routes[best_index], fitness[best_index]  # 精英不用参与自然选择，要和下一代的胜者比较

# 模拟遗传
not_improve_time = 0
for i in range(generations):
    routes = selection(routes, fitness)  # 选择
    routes = crossover(routes, len(cities))  # 交叉
    routes = mutation(routes, len(cities))  # 变异
    fitness = get_all_routes_fitness(routes, dist_matrix)
    #argmax返回max的索引
    best_route_index = fitness.argmax()
    if fitness[best_route_index] > best_fitness:
        not_improve_time = 0
        best_route, best_fitness = routes[best_route_index], fitness[best_route_index]  # 保存最优路线及其适应度
    else:
        not_improve_time += 1

    
    if (i + 1) % 100 == 0:
        print('generations: {}, 当前最优路线距离： {}'.format(i + 1, 1 / best_fitness)) #1/fitness还原路径距离
    if not_improve_time >= 3000:
        print('3000代都没有出现更短的路径,结束，返回问题的最好近似解')
        break
x=[]
y=[]
for i in best_route :
    (xi,yi)=cities[i]
    x.append(xi)
    y.append(yi)
(x0,y0)=cities[best_route[0]]
x.append(x0)
y.append(y0)
for i in range (0,len(best_route)+1) :
    plt.scatter(x,y,c='b')
    plt.plot(x,y,c='r')
plt.show()

print('最优路线为：')
print(best_route)
print('总距离为： {}'.format(1 / best_fitness))
