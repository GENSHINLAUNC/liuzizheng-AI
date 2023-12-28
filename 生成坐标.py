import numpy as np

def generate_unique_axis(num_points):
    # 生成均不相等的二维坐标
    x_axis = np.random.choice(np.arange(10000), size=num_points, replace=False)
    y_axis = np.random.choice(np.arange(10000), size=num_points, replace=False)

    # 将 x 和 y 坐标合并成二维数组
    axis = np.column_stack((x_axis, y_axis))

    return axis

# 生成一百个均不相等的坐标
unique_axis = generate_unique_axis(100)

# 输出格式化结果
for i, (x, y) in enumerate(unique_axis, start=0):
    print (f"{i} {x} {y}")
