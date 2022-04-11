from datetime import time
from tkinter import *
import tkinter.filedialog
from tkinter.ttk import Combobox
from timeit import default_timer as timer
import time
import xlrd
import pymysql
import numpy as np
from matplotlib import pyplot as plt


LOG_NUM = 0

toop = Tk()
toop.title("软件工程")  # 定义标题
toop.geometry("700x550+200+50")
toop.configure(bg = 'lightskyblue')
frame0 = Frame(toop,height = 50) # 标题
frame1 = Frame(toop,height = 100) # 选择框架
frame2 = Frame(toop,width=700, height=100) # 功能框架
frame3 = Frame(toop,width=400, height=300) # 输出框架
frame4 = Frame(toop,width=400, height=300) # 日志框架
frame5 = Frame(toop,width=100, height=200) # 辅助功能框架
frame0.grid(row=0,  padx=50)
frame1.grid(row=1,  padx=20,pady=10)
frame2.grid(row=2,  padx=50)
frame3.grid(row=3,  padx=20)
frame4.grid(row=4,  padx=20)
frame5.grid(row=5,  padx=20)
frame5.configure(bg = 'lightskyblue')
frame0.configure(bg = 'lightskyblue')
frame1.configure(bg = 'lightskyblue')
frame2.configure(bg = 'lightskyblue')
frame3.configure(bg = 'lightskyblue')
frame4.configure(bg = 'lightskyblue')

path = StringVar() # 地址存储
book = StringVar()

def xz():
    m=tkinter.filedialog.askopenfilename()# 设置变量m的值
    if path != '':
         path.set(m)
    return path

# 标题框架
name = Label(frame0, text='{0-1}KP实例数据集算法实验平台', font=('华文隶书', 19), fg='black',bg = 'lightskyblue')
name.grid( pady=15)

# 选择框架
Entry(frame1, width=50,textvariable=path).grid(row=0, column=0, padx=10,pady=10)
btn1 = Button(frame1,text='请选择文件',command=xz)
btn1.grid(row=0, column=1, padx=10, pady=5)
cv = tkinter.StringVar()
com = Combobox(frame1, textvariable=cv)
com.grid(row=0, column=2, padx=10, pady=5)
com["value"] = ("请选择算法","贪心算法", "动态规划法", "回溯法", "遗传算法")
com.current(0)

# 数据初始化
def Initial():
    f = open(path.get(), 'r')
    num = f.readlines()[1:]
    num = [line.strip("\n") for line in num]
    f1 = open(path.get(), 'r')
    tnum = f1.readlines()[:1]
    tnum = [line.strip("\n") for line in tnum]
    f.close()
    f1.close()
    print(num)
    print(tnum)
    return num, tnum


def painter():
    resx = Initial()
    x, y = np.loadtxt(resx[0], delimiter=' ', unpack=True)
    plt.plot(x, y, '.', color='red')
    plt.xlabel('wight')
    plt.ylabel('value')
    plt.title('scatter plot')
    plt.legend()
    pr_log_text("绘制散点图成功")
    plt.show()


# 贪心算法：
def GreedyAlgo(item, c, num):
    data = np.array(item)
    index = np.lexsort([data[:, 0], -1 * data[:, 1]])
    status = [0] * num
    Tw = 0
    Tv = 0

    for i in range(num):
        if data[index[i], 0] <= c:
            Tw += data[index[i], 0]
            Tv += data[index[i], 1]
            status[index[i]] = 1
            c -= data[index[i], 0]
        else:
            continue

    print("贪心算法，最大价值为：")
    return Tv


# 动态规划算法
def Dp(w, v, c, num):
    cnt = [0 for j in range(c + 1)]

    for i in range(1, num + 1):
        for j in range(c, 0, -1):
            if j >= w[i - 2]:
                cnt[j] = max(cnt[j], cnt[j - w[i - 2]] + v[i - 2])

    print("动态规划算法，最大价值为：")
    return cnt[c]


# 回溯法
curV = 0
curW = 0
bestV = 0
bestW = 0


def Backtracking(k, c, num):
    m = Initial()
    a0 = []
    for line in m[0]:
        line = line.split(' ')  # 以空格划分
        a0.append(line)
    a = np.array(a0)
    a = a.astype(int)
    w = (a[:, 0])
    v = (a[:, 1])
    w = np.array(w)
    v = np.array(v)
    w = w.astype(int)
    v = v.astype(int)
    global curW, curV, bestV, bestW
    status = [0 for i in range(num)]

    if k >= num:
        if bestV < curV:
            bestV = curV
    else:
        if curW + w[k] <= c:
            status[k] = 1
            curW += w[k]
            curV += v[k]
            Backtracking(k + 1, c, num)
            curW -= w[k]
            curV -= v[k]
        status[k] = 0
        Backtracking(k + 1, c, num)

N = 500     ##迭代次数
Pc = 0.8    ##交配概率
Pm = 0.15   ##变异概率


## 遗传算法：
# 初始化,N为种群规模，n为染色体长度
def init(N,n):
    C = []
    for i in range(N):
        c = []
        for j in range(n):
            a = np.random.randint(0,2)
            c.append(a)
        C.append(c)
    return C


def fitness(C,N,n,W,V,w):
    S = []##用于存储被选中的下标
    F = []## 用于存放当前该个体的最大价值
    for i in range(N):
        s = []
        h = 0  # 重量
        f = 0  # 价值
        for j in range(n):
            if C[i][j]==1:
                if h+W[j]<=w:
                    h=h+W[j]
                    f = f+V[j]
                    s.append(j)
        S.append(s)
        F.append(f)
    return S,F


##适应值函数,B位返回的种族的基因下标，y为返回的最大值
def best_x(F,S,N):
    y = 0
    x = 0
    B = [0]*N
    for i in range(N):
        if y<F[i]:
            x = i
        y = F[x]
        B = S[x]
    return B,y


## 计算比率
def rate(x):
    p = [0] * len(x)
    s = 0
    for i in x:
        s += i
    for i in range(len(x)):
        p[i] = x[i] / s
    return p

## 选择
def chose(p, X, m, n):
    X1 = X
    r = np.random.rand(m)
    for i in range(m):
        k = 0
        for j in range(n):
            k = k + p[j]
            if r[i] <= k:
                X1[i] = X[j]
                break
    return X1


##交配
def match(X, m, n, p):
    r = np.random.rand(m)
    k = [0] * m
    for i in range(m):
        if r[i] < p:
            k[i] = 1
    u = v = 0
    k[0] = k[0] = 0
    for i in range(m):
        if k[i]:
            if k[u] == 0:
                u = i
            elif k[v] == 0:
                v = i
        if k[u] and k[v]:
            # print(u,v)
            q = np.random.randint(n - 1)
            # print(q)
            for i in range(q + 1, n):
                X[u][i], X[v][i] = X[v][i], X[u][i]
            k[u] = 0
            k[v] = 0
    return X


##变异
def vari(X, m, n, p):
    for i in range(m):
        for j in range(n):
            q = np.random.rand()
            if q < p:
                X[i][j] = np.random.randint(0,2)

    return X

def xs():
    a0 = []
    a1 = []
    m = Initial()
    for line in m[0]:
        line = line.split(' ')  # 以空格划分
        a0.append(line)
    for line in m[1]:
        line = line.split(' ')
        a1.append(line)

    c = int(a1[0][0])
    num = int(a1[0][1])

    a = np.array(a0)
    a = a.astype(int)
    w = (a[:, 0])
    v = (a[:, 1])
    w = np.array(w)
    v = np.array(v)
    w = w.astype(int)
    v = v.astype(int)
    item = list(zip(w, v))

# 排序
def sort():
    a0 = []
    descending = []
    # 计算重量与价值的比值
    m = Initial()
    for line in m[0]:
        line = line.split(' ')
        a0.append(line)
    for i in range(len(a0)):
        F0 = int(a0[i][0])
        S0 = int(a0[i][1])
        T0 = F0 / S0
        m = round(T0,3)
        descending.append(m)
    descending.sort(reverse=True)
    print(descending)
    for item in descending:
        px.insert(END, item)
    pr_log_text("按照重量/质量非递增排列成功")


# 日志动态打印：
def pr_log_text(logmsg):
    global LOG_NUM
    current_time = obtain_time()
    logmsg_in = str(current_time) + " " + str(logmsg) + "\n"  # 换行
    if LOG_NUM <= 7:
        log_txt.insert(END, logmsg_in)
        LOG_NUM = LOG_NUM + 1
    else:
        log_txt.delete(1.0, 2.0)
        log_txt.insert(END, logmsg_in)

def xz1():
    m=tkinter.filedialog.askopenfilename()# 设置变量m的值
    if book != '':
         book.set(m)
    sheet = book.sheet_by_name("Sheet1")
    #建立一个MySQL连接
    conn = pymysql.connect(
            host='localhost',
            user='root',
            passwd='',
            db='mydata',
            port=3306,
            charset='utf8'
            )
    # 获得游标
    cur = conn.cursor()
    # 创建插入SQL语句
    query = 'insert into beibao0 (Weight,Value,Proportion) values (%s, %s, %s)'
    # 创建一个for循环迭代读取xls文件每行数据的, 从第二行开始是要跳过标题行
    for r in range(1, sheet.nrows):
          Weight = sheet.cell(r,0).value
          Value = sheet.cell(r,1).value
          Proportion = sheet.cell(r, 2).value
          values = (Weight,Value,Proportion)
          # 执行sql语句
          cur.execute(query, values)
    cur.close()
    conn.commit()
    conn.close()
    columns = str(sheet.ncols)
    rows = str(sheet.nrows)
    print ("导入 " +columns + " 列 " + rows + " 行数据到MySQL数据库!")

# 获取当前时间
def obtain_time():
    current_time = time.strftime('%Y/%m/%d %H:%M:%S', time.localtime(time.time()))
    return current_time
def clear():
    result_text.delete(1.0,END)
    log_txt.delete(1.0,END)
    px.delete(0,END)
    

def choose():  # 处理事件
    a0 = []
    a1 = []
    m = Initial()
    for line in m[0]:
        line = line.split(' ')  # 以空格划分
        a0.append(line)
    for line in m[1]:
        line = line.split(' ')
        a1.append(line)

    c = int(a1[0][0])
    num = int(a1[0][1])

    a = np.array(a0)
    a = a.astype(int)
    w = (a[:, 0])
    v = (a[:, 1])
    w = np.array(w)
    v = np.array(v)
    w = w.astype(int)
    v = v.astype(int)
    item = list(zip(w, v))

    if com.get() == "贪心算法":
        time_start = timer()
        ret1 = GreedyAlgo(item, c, num)
        print(ret1)
        time_end = timer()
        time_sum = time_end - time_start
        print("求解时间为：")
        print(time_sum)
        result = "贪心算法：最优解：" + str(ret1) + "，" + "求解时间：" + str(time_sum)
        file = open('D:\软件工程\实验二\测试数据\\zyx.txt', 'w')
        file.write(result)
        file.close()
        result_text.delete(1.0, END)
        result_text.insert(1.0, '最优解为：')
        result_text.insert(1.8, ret1)
        result_text.insert(2.0, '\n')
        result_text.insert(2.0, '运行时间为：')
        result_text.insert(2.8, time_sum)
        pr_log_text("使用贪心算法求解成功")
    elif com.get() == "动态规划算法":
        time_start = timer()
        ret2 = Dp(w, v, c, num)
        print(ret2)
        time_end = timer()
        time_sum = time_end - time_start
        print("求解时间为：")
        print(time_sum)
        result = "动态规划算法：最优解：" + str(ret2) + "，" + "求解时间：" + str(time_sum)
        file = open('D:\软件工程\实验二\测试数据\\zyx.txt', 'w')
        file.write(result)
        file.close()
        result_text.delete(1.0, END)
        result_text.insert(1.0, '最优解为：')
        result_text.insert(1.8, ret2)
        result_text.insert(2.0, '\n')
        result_text.insert(2.0, '运行时间为：')
        result_text.insert(2.8, time_sum)
        pr_log_text("使用动态规划算法求解成功")
    elif com.get() == "回溯法":
        time_start = timer()
        Backtracking(0, c, num)
        print(bestV)
        time_end = timer()
        time_sum = time_end - time_start
        print("求解时间为：")
        print(time_sum)
        result = "回溯算法：最优解：" + str(bestV) + "，" + "求解时间：" + str(time_sum)
        file = open('D:\软件工程\实验二\测试数据\\zyx.txt', 'w')
        file.write(result)
        file.close()
        result_text.delete(1.0, END)
        result_text.insert(1.8, bestV)
        result_text.insert(2.0, '\n')
        result_text.insert(2.0, '运行时间为：')
        result_text.insert(2.8, time_sum)
        pr_log_text("使用回溯法求解成功")
    elif com.get() == "遗传算法":
        time_start = timer()
        n = len(w)
        C = init(num, n)
        S, F = fitness(C, num, n, w, v, c)
        B, y = best_x(F, S, num)
        Y = [y]
        for i in range(N):
            p = rate(F)
            C = chose(p, C, num, n)
            C = match(C, num, n, Pc)
            C = vari(C, num, n, Pm)
            S, F = fitness(C, num, n, w, v, c)
            B1, y1 = best_x(F, S, num)
            if y1 > y:
                y = y1
            Y.append(y)
        print("遗传算法，最大价值为：")
        print(y)
        time_end = timer()
        time_sum = time_end - time_start
        print("求解时间为：")
        print(time_sum)
        result = "遗传算法：最优解：" + str(bestV) + "，" + "求解时间：" + str(time_sum)
        file = open('D:\软件工程\实验二\测试数据\\zyx.txt', 'w')
        file.write(result)
        file.close()
        result_text.delete(1.0, END)
        result_text.insert(1.0, '最优解为：')
        result_text.insert(1.8, y)
        result_text.insert(2.0, '\n')
        result_text.insert(2.0, '运行时间为：')
        result_text.insert(2.8, time_sum)
        pr_log_text("使用遗传法求解成功")

# 功能框架
btn2 = Button(frame2,text='求解问题',command=choose, width=15, height=1)
btn2.grid(row=0, column=0, padx=20, pady=10)
btn3 = Button(frame2,text='绘制散点图',command=painter, width=15, height=1)
btn3.grid(row=0, column=3, padx=20, pady=10)
btn4 = Button(frame2,text='重量比排序',command=sort, width=15, height=1)
btn4.grid(row=0, column=5, padx=20, pady=10)
btn5 = Button(frame2,text='上传数据',command=xz1, width=15, height=1)
btn5.grid(row=0, column=7, padx=20, pady=10)


# 结果框架
Label(frame3, text="结果：",bg = 'lightskyblue').grid(row=0, column=0)
result_text = Text(frame3, width=90, height=3)  # 结果输出框
result_text.grid(row=1, column=0, columnspan=30, padx=8)
Label(frame3, text="排序结果：",bg = 'lightskyblue').grid(row=2, column=0)
px = Listbox(frame3, width=90, height=5)  # 日志输出框
px.grid(row=3, column=0, columnspan=30, padx=8)
Label(frame4, text="日志: ",bg = 'lightskyblue').grid(row=0, column=0)
log_txt = Text(frame4, width=90, height=7)  # 日志输出框
log_txt.grid(row=1, column=0, columnspan=30, padx=8)

# 退出框架
btn6 = Button(frame5,text='清空',command=clear, width=15, height=1)
btn6.grid(row=0, column=0, padx=20, pady=10)
btn7 = Button(frame5,text='退出',command=toop.quit, width=15, height=1)
btn7.grid(row=0, column=1, padx=20, pady=10)
toop.mainloop()
