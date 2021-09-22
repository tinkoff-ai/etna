import numpy as np

def SSE(i, j, p, pp): #считается ошибка аппроксимации с i по j элемент константой (среднее значение)
    if i == 0:
        avg = p[j]
        return pp[j] - avg ** 2 / (j - i + 1)
    avg = p[j] - p[i-1]
    return pp[j] - pp[i-1] - avg ** 2 / (j - i + 1)

def v_optimal_hist(series, B): #считается ошибка аппроксиации ряда series с B бинов
    p, pp = np.empty_like(series), np.empty_like(series)
    p[0] = series[0] #p[i] = series[0] + series[1] + .. + series[i]
    pp[0] = series[0] ** 2 #pp[i] = series[0]**2 + series[1]**2 + .. + series[i]**2
    for i in range(1, len(series)):
        p[i] = p[i-1] + series[i]
        pp[i] = pp[i-1] + series[i] ** 2
    
    sse = np.zeros((len(series), B)) # sse[i][j] = ошибка аппроксимации j+1 бинами ряда series[:i+1]
    for i in range(len(series)): #заполняем столбец матрицы для 1 бина
        sse[i][0] = SSE(0, i, p, pp) 
    
    count_sse = np.zeros((len(series), len(series))) # count_sse[i][j] = ошибка аппроксимации с 1 бином от series[i:j+1]
    for i in range(len(series)): #препдосчитываем для того чтобы тысячу раз не вызывать одно и то же
        for j in range(i, len(series)):
            count_sse[i][j] = SSE(i, j, p, pp)
    
    #начинаем заполнять sse
    for k in range(1, B): #итерация по бинам
        for i in range(k, len(series)): #итерация по ряду
            now_min = np.inf
            for j in range(0, i):
                now = sse[j][k-1] + count_sse[j+1][i]
                if now < now_min:
                    now_min = now
            sse[i][k] = now_min
    return sse[len(series)-1][B-1]