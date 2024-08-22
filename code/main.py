import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tkinter as tk
import tkinter.ttk as ttk
from PIL import ImageTk, Image
from matplotlib import rcParams
from sklearn.preprocessing import MinMaxScaler
import os
import sys
os.chdir(os.path.dirname(os.path.abspath(sys.argv[0])))
x_values = [0.02, 0.1, 0.3, 0.5, 6, 10, 15, 20, 45, 60, 100, 150, 300, 340, 350, 455, 500,
            800, 1000, 1200, 2300, 3300, 4000, 5000, 9000, 10400, 15000, 50000, 100000,
            6000, 7000, 8000, 10000, 11000, 12000, 13000, 14000, 16000, 19000, 21000, 24000,
            27000, 30000, 35000, 40000, 45000, 56000, 65000, 71000, 77000, 82000, 88000, 94000]

y_values = [[0.01, 0.02, 0.06, 0.12, 0.14, 0.32, 0.43],
            [0.02, 0.03, 0.1, 0.21, 0.23, 0.55, 0.56],
            [0.02, 0.05, 0.15, 0.31, 0.34, 0.79, 0.68],
            [0.03, 0.06, 0.17, 0.36, 0.4, 0.74, 0.93],
            [0.06, 0.16, 0.4, 0.83, 0.12, 2.14, 1.14],
            [0.07, 0.2, 0.47, 0.99, 1.41, 2.53, 1.25],
            [0.08, 0.23, 0.54, 1.13, 1.68, 2.9, 1.34],
            [0.09, 0.26, 0.59, 1.24, 1.91, 3.19, 1.41],
            [0.12, 0.41, 0.77, 1.63, 2.74, 4.18, 1.61],
            [0.13, 0.41, 0.85, 1.79, 3.11, 4.61, 1.68],
            [0.15, 0.5, 1.01, 2.12, 3.9, 5.46, 1.82],
            [0.18, 0.59, 1.16, 2.43, 4.67, 6.25, 1.94],
            [0.22, 0.78, 1.46, 3.06, 6.33, 7.88, 2.14],
            [0.23, 0.82, 1.52, 3.2, 6.68, 8.21, 2.18],
            [0.23, 0.83, 1.53, 3.23, 6.77, 8.29, 2.19],
            [0.25, 0.92, 1.67, 3.52, 7.59, 9.05, 2.26],
            [0.26, 0.95, 1.73, 3.63, 7.91, 9.34, 2.29],
            [0.31, 1.15, 2.02, 4.25, 9.7, 10.9, 2.43],
            [0.33, 1.26, 2.18, 4.58, 10.7, 11.8, 2.5],
            [0.35, 1.35, 2.31, 4.86, 11.6, 12.5, 2.56],
            [0.44, 1.75, 2.87, 6.04, 15.3, 15.5, 2.77],
            [0.49, 2.02, 3.24, 6.82, 17.5, 17.9, 2.89],
            [0.53, 2.19, 3.45, 7.27, 19.4, 18.7, 2.96],
            [0.57, 2.39, 3.72, 7.83, 21.3, 20.1, 3.05],
            [0.69, 3.02, 4.53, 9.52, 27.4, 24.5, 3.32],
            [0.72, 3.2, 4.75, 9.99, 29.1, 25.7, 3.4],
            [0.82, 3.71, 5.37, 11.3, 34, 29, 3.63],
            [1.22, 6.01, 8.02, 16.9, 51.4, 43.3, 5.05],
            [1.54, 7.92, 10.1, 21.2, 64.2, 54.6, 6.99],
             [0.6, 2.57, 3.95, 8.32, 23.1, 21.4, 3.51],
            [0.63, 2.74, 4.16, 8.76, 24.6, 22.5, 3.58],
            [0.66, 2.89, 4.35, 9.16, 26.1, 23.5, 3.64],
            [0.71, 3.15, 4.69, 9.86, 28.6, 25.3, 3.76],
            [0.74, 3.28, 4.84, 10.2, 29.8, 26.2, 3.81],
            [0.76, 3.39, 4.98, 10.5, 31, 26.9, 3.87],
            [0.78, 3.5, 5.12, 10.8, 32, 27.7, 3.92],
            [0.8, 3.61, 5.25, 11, 33, 28.4, 3.96],
            [0.83, 3.81, 5.48, 11.5, 35, 29.6, 4.06],
            [0.88, 4.08, 5.81, 12.2, 37.6, 31.4, 4.19],
            [0.91, 4.24, 6, 12.6, 39, 32.5, 4.28],
            [0.96, 4.48, 6.28, 13.2, 40.7, 33.9, 4.4],
            [0.99, 4.69, 6.53, 13.7, 42.2, 35.3, 4.53],
            [1.03, 4.9, 6.76, 14.2, 43.7, 36.6, 4.65],
            [1.08, 5.21, 7.12, 15, 45.8, 38.5, 4.84],
            [1.13, 5.49, 7.44, 15.7, 47.8, 40.2, 5.03],
            [1.18, 5.76, 7.74, 16.3, 49.7, 41.8, 5.22],
            [1.27, 6.28, 8.33, 17.5, 53.3, 45, 5.63],
            [1.33, 6.67, 8.75, 18.4, 55.9, 47.3, 5.97],
            [1.37, 6.91, 9.01, 19, 57.5, 48.7, 6.19],
            [1.41, 7.14, 9.26, 19.5, 59, 50.1, 6.41],
            [1.44, 7.32, 9.45, 19.9, 60.2, 51.1, 6.6],
            [1.47, 7.53, 9.68, 20.4, 61.6, 52.3, 6.82],
            [1.51, 7.73, 9.9, 20.8, 62.9, 53.5, 7.05]]

# 데이터 준비: x_values, y_values
x_train = np.array(x_values).reshape(-1, 1)
y_train = np.array(y_values)

# 데이터 스케일링 적용.
scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()

x_train = scaler_x.fit_transform(x_train)
y_train = scaler_y.fit_transform(y_train)

# 딥러닝 모델 정의.
model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, input_shape=(1,), activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(7, activation='linear')  # 출력 유닛을 7개로 설정합니다.
])

# 모델 훈련 구성.
model.compile(loss='mean_squared_error',
              optimizer=tf.keras.optimizers.Adam(0.01))

# 모델 학습.
model.fit(x_train, y_train, epochs=1000, verbose=0, validation_split=0.2)

# 한글 폰트 설정
rcParams['font.family'] = 'NanumGothic'

#그래프 색 변경
plt.style.use('dark_background')
plt.rcParams['text.color'] = 'white'
plt.rcParams['axes.labelcolor'] = 'white'
plt.rcParams['xtick.color'] = 'white'
plt.rcParams['ytick.color'] = 'white'

#변수 선언
Dbg="black"
Dfg="white"
Dfont1=("나눔고딕",15,"bold")
Dfont2=("나눔고딕",13,"bold")
Dfont3=("나눔고딕",11,"bold")
name= [
    "Davy Crockett","Crude nuclear terrorist weapon","B-61 Mod 3","North Korean weapon tested in 2006","North Korean weapon tested in 2009","North Korean weapon tested in 2013",
    "Improvised HEU device","Little Boy","Gadget","Fat man","Largest Pakistani Weapon Tested","Largest Indian Weapon Tested",
    "W-76","North Korean Weapon Tested in 2017","W-80","W-87","TN 80/81","B-61 mob7",
    "W-78","W-88","Ivy King","Topol","W-59","B-83",
    "R-12","Dong Feng-4","W-39","Dong Feng-5","W-53","Ivy Mike",
    "Castle Bravo","Tsar Bomba50","Tsar Bomba100","커스텀"
]
missile_data = [
    [0.02, 0.01, 0.02, 0.06, 0.12, 0.14, 0.32, 0.43],
    [0.1, 0.02, 0.3, 0.1, 0.21, 0.23, 0.55, 0.56],
    [0.3, 0.02, 0.05, 0.15, 0.31, 0.34, 0.79, 0.68],
    [0.5, 0, 0.05, 0, 0.56, 0.41, 1.57, 0.7],
    [6, 0, 0.12, 0, 1.28, 1.2, 3.59, 0.99],
    [10, 0, 0.15, 0, 1.51, 1.53, 4.26, 1.05],
    [10, 0.07, 0.2, 0.47, 0.99, 1.41, 2.53, 1.25],
    [15, 0, 0.18, 0.34, 1.67, 1.91, 4.52, 1.2],
    [20, 0.09, 0.2, 0.6, 1.27, 1.41, 2.27, 3.27],
    [20, 0, 0.2, 0.76, 1.72, 2.21, 4.59, 1.31],
    [45, 0, 0.28, 0, 2.5, 3.05, 7.03, 1.16],
    [60, 0, 0.31, 0, 2.75, 3.48, 7.74, 1.16],
    [100, 0, 0.38, 0, 3.26, 4.38, 9.18, 1.11],
    [150, 0, 0.45, 0, 3.74, 5.26, 10.5, 1],
    [150, 0, 0.45, 0, 3.74, 5.26, 10.5, 1],
    [300, 0, 0.6, 0, 4.71, 7.17, 13.2, 0.46],
    [300, 0, 0.6, 0, 4.71, 7.17, 13.2, 0.46],
    [340, 0, 0.63, 0, 4.91, 7.58, 13.8, 0],
    [350, 0, 0.63, 0, 4.95, 7.67, 13.9, 0],
    [455, 0, 0.71, 0, 5.41, 8.62, 15.2, 0],
    [500, 0.26, 0.95, 1.73, 3.63, 7.91, 9.34, 2.29],
    [800, 0, 0.88, 0, 6.53, 11.1, 18.4, 0],
    [1000, 0, 0.97, 0, 7.03, 12.2, 19.8, 0],
    [1200, 0, 1.04, 0, 7.47, 13.2, 21, 0],
    [2300, 0.01, 0.02, 0.06, 0.12, 0.14, 0.32, 0.43],
    [3300, 0, 1.56, 0, 10.5, 20.5, 29.4, 0],
    [4000, 0.53, 2.19, 3.45, 7.27, 1.94, 1.87, 2.96],
    [5000, 0, 1.84, 0, 12, 24.5, 33.8, 0],
    [9000, 0, 1.84, 0, 12, 24.5, 33.8, 0],
    [10400, 0.72, 3.2, 4.75, 9.99, 29.1, 25.7, 3.4],
    [15000, 0.82, 3.71, 5.37, 11.3, 34, 29, 3.63],
    [50000, 0, 4.62, 8.91, 20.7, 54.3, 60, 3.14],
    [100000, 0, 6.1, 0, 32.6, 73, 91.8, 0]
]
missile_cont=[
    "미국", "(국가 불특정)", "미국",
    "북한", "북한", "북한",
    "(국가 불특정)", "미국",
    "미국", "미국", "파키스탄",
    "인도", "미국", "북한",
    "미국", "미국", "프랑스",
    "미국", "미국", "미국",
    "미국", "러시아", "미국",
    "소련 (현 러시아)", "중국", "미국",
    "중국", "미국", "미국",
    "미국", "미국", "소련 (현 러시아)",
    "소련 (현 러시아)","한국"]
population,wide,densities=0.0,0.0,0.0 #인구수, 넓이, 인구밀도
snames,scont,skt=0,"",0.0   #선택된 핵 이름, 국가, kt양
ar,fr,hr,mr,tr,lr,rr=0.0,0.0,0.0,0.0,0.0,0.0,0.0
dpeople,darea=0.0,0.0 #피해자, 피해거리
#창 선언 main
main = tk.Tk()
main.title("핵미사일 시뮬레이터")
#main.geometry("800x1000")
main.resizable(False, False)
main.configure(bg="black")

button=tk.Toplevel(main)
button.title("발사버튼")
button.resizable(False,False)
button.configure(bg="black")
#함수
def G_dataset():    #artiW
    plt.close()
    plt.figure(num='데이터 셋 표', figsize=(8, 6))
    # 데이터 전치
    transposed_data = np.transpose(missile_data)

    # y 데이터 추출
    y_values = transposed_data[0]

    # x 데이터 추출
    x_values = transposed_data[1:]

    # x축 라벨 설정
    labeel=['공기 폭발 반경', '화염구 반경', '강한 폭발 피해 반경(20psi)', '보통 폭발 피해 반경', '열 복사 반경 (3도 화상)', '약한 폭발 피해 반경 (1 psi)', '방사선 반경 (500 rem)']

    # 그래프 색상
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'w']

    # 그래프 그리기
    for i, x_row in enumerate(x_values):
        plt.plot(x_row, y_values, 'o', color=colors[i], label=labeel[i])

    # 그래프 스타일 설정
    plt.xlabel('피해(km²)')
    plt.ylabel('우라늄양(KT)')
    plt.title('핵 미사일 데이터')
    plt.legend()
    plt.grid(True)

    # 그래프 보여주기
    plt.show()

def W_info():
    #창 선언 infoW
    infoW=tk.Toplevel(main)
    infoW.title("프로그램 정보")
    infoW.configure(bg="black")

    #위젯 선언 infoW
    infoWla01=tk.Label(infoW,text="프로그램 정보",font=("나눔고딕",25,"bold"),bg=Dbg,fg=Dfg)
    infoWla02=tk.Label(infoW,text="참여자",font=Dfont1,bg=Dbg,fg=Dfg)
    infoWla03=tk.Label(infoW,text="31203 김도경 : 프로그램 제작, 자료조사/분석, 보고서 작성",font=Dfont3,bg=Dbg,fg=Dfg,anchor="w")
    infoWla04=tk.Label(infoW,text="31207 문지환 : 프로그램 제작 보조, 자료조사/분석 ",font=Dfont3,bg=Dbg,fg=Dfg,anchor="w")
    infoWla05=tk.Label(infoW,text="31219 이장현 : 프록그램 제작 보조, 자료조사/분석 ",font=Dfont3,bg=Dbg,fg=Dfg,anchor="w")
    infoWla06=tk.Label(infoW,text="31222 천인희 : 자료조사/분석, 시스템UI 이미지 제작",font=Dfont3,bg=Dbg,fg=Dfg,anchor="w")
    infoWla07=tk.Label(infoW,text="흥진고\n2023",font=Dfont1,bg=Dbg,fg=Dfg)
    img=ImageTk.PhotoImage(Image.open("..\\image\\HJ.png"))
    infoWla08=tk.Label(infoW,image=img,bg=Dbg,fg=Dfg,anchor="center")

    infoWlab1=tk.Label(infoW,font=("나눔고딕",3))
    infoWlab2=tk.Label(infoW,font=("나눔고딕",3))

    #위젯 배치 infoW
    infoWla01.grid(row=0,column=0,sticky="nsew")
    infoWlab1.grid(row=1,column=0,columnspan=3,sticky="nsew")
    infoWla02.grid(row=2,column=0,rowspan=4,sticky="nsew")
    infoWla03.grid(row=2,column=1,sticky="nsew")
    infoWla04.grid(row=3,column=1,sticky="nsew")
    infoWla05.grid(row=4,column=1,sticky="nsew")
    infoWla06.grid(row=5,column=1,sticky="nsew")
    infoWlab2.grid(row=6,column=0,columnspan=2,sticky="nsew")
    infoWla07.grid(row=7,column=0,sticky="nsew")
    infoWla08.grid(row=7,column=1,sticky="nsew")
    infoWla07.image = img

def T_dataset():    #artiW
    #창 선언    dataT

    dataT = tk.Tk()
    dataT.title("데이터셋 표")
    dataT.configure(bg="black")
    
    # 표의 행(row) 개수와 열(column) 개수
    rows = len(missile_data) + 1
    columns = 9
    labeel = ['핵 미사일 이름', '우라늄양(KT)', '공기 폭발 반경(KM)', '화염구 반경(KM)', '강한 폭발 피해 반경(20psi/KM)',
              '보통 폭발 피해 반경(KM)', '열 복사 반경 (3도 화상/KM)', '약한 폭발 피해 반경 (1 psi/KM)', '방사선 반경 (500 rem/KM)']

    # 표를 생성하고 배치
    for i in range(rows):
        for j in range(columns):
            if i == 0:  # 첫 번째 행은 열 제목
                label = tk.Label(dataT, text=labeel[j], bg=Dbg, fg=Dfg,font=Dfont3)
            else:
                if j == 0:  # 첫 번째 열은 행 제목
                    label = tk.Label(dataT, text=name[i - 1], bg=Dbg, fg=Dfg,font=Dfont3)
                else:
                    label = tk.Label(dataT, text=missile_data[i - 1][j - 1], bg=Dbg, fg=Dfg,font=Dfont3)
            label.grid(row=i, column=j)

    dataT.mainloop()

def W_arti():
    #창 선언 artiW
    artiW=tk.Toplevel(main)
    artiW.title("인공신경망 정보")
    artiW.configure(bg="black")

    #위젯 선언 artiW
    artiWla01=tk.Label(artiW,text="인공신경망 정보",font=("나눔고딕",25,"bold"),bg=Dbg,fg=Dfg)
    artiWla02=tk.Label(artiW,text="사용된 데이터셋",font=Dfont1,bg=Dbg,fg=Dfg,anchor="w")
    artiWbu01=tk.Button(artiW,text="그래프 보기",font=Dfont3,bg=Dbg,fg=Dfg,command=G_dataset)
    artiWbu02=tk.Button(artiW,text="표 보기",font=Dfont3,bg=Dbg,fg=Dfg,command=T_dataset)
    artiWla03=tk.Label(artiW,text="딥러닝 모델 정보",font=Dfont1,bg=Dbg,fg=Dfg)
    artiWla04=tk.Label(artiW,text="활성화 함수",font=Dfont3,bg=Dbg,fg=Dfg,anchor="w")
    artiWla05=tk.Label(artiW,text="relu",font=Dfont3,bg=Dbg,fg=Dfg,anchor="e")
    artiWla06=tk.Label(artiW,text="구조",font=Dfont3,bg=Dbg,fg=Dfg,anchor="w")
    artiWla07=tk.Label(artiW,text="다중 퍼셉트론(MLP)",font=Dfont3,bg=Dbg,fg=Dfg,anchor="e")
    artiWla08=tk.Label(artiW,text="딥러닝 모델 함수",font=Dfont3,bg=Dbg,fg=Dfg,anchor="w")
    artiWbu03=tk.Button(artiW,text="그래프 보기",font=Dfont3,bg=Dbg,fg=Dfg,command=G_func)

    artiWlab1=tk.Label(artiW,font=("나눔고딕",3))
    artiWlab2=tk.Label(artiW,font=("나눔고딕",3))

    #위젯 배치 artiW
    artiWla01.grid(row=0,column=0,sticky="nsew")
    artiWlab1.grid(row=1,column=0,columnspan=3,sticky="nsew")
    artiWla02.grid(row=2,column=0,sticky="nsew")
    artiWbu01.grid(row=2,column=1,sticky="nsew")
    artiWbu02.grid(row=2,column=2,sticky="nsew")
    artiWlab2.grid(row=3,column=0,columnspan=3,sticky="nsew")
    artiWla03.grid(row=4,column=0,rowspan=3,sticky="nsew")
    artiWla04.grid(row=4,column=1,sticky="nsew")
    artiWla05.grid(row=4,column=2,sticky="nsew")
    artiWla06.grid(row=5,column=1,sticky="nsew")
    artiWla07.grid(row=5,column=2,sticky="nsew")
    artiWla08.grid(row=6,column=1,sticky="nsew")
    artiWbu03.grid(row=6,column=2,sticky="nsew")

def W_sett():
    def enter01(event):
        global population,wide,densities
        population=round(float(eval(settWen01.get())),3)
        la18.config(text=population)
        if(wide!=0):
            densities=round(population/wide,3)
            la11.config(text=densities)

    def enter02(event):
        global population,wide,densities
        wide=round(float(eval(settWen02.get())),3)
        la09.config(text=wide)
        if(population!=0):
            densities=round(population/wide,3)
            la11.config(text=densities)

    settW=tk.Toplevel(main)
    settW.title("발사 환경 설정")
    settW.configure(bg="black")

    #위젯 선언 artiW
    settWla01=tk.Label(settW,text="발사 환경 설정",font=("나눔고딕",25,"bold"),bg=Dbg,fg=Dfg)
    settWla02=tk.Label(settW,text="인구수",font=Dfont3,bg=Dbg,fg=Dfg,anchor="w")
    settWen01=tk.Entry(settW,font=Dfont3,bg=Dbg,fg=Dfg)
    settWla03=tk.Label(settW,text="넓이(km²)",font=Dfont3,bg=Dbg,fg=Dfg,anchor="w")
    settWen02=tk.Entry(settW,font=Dfont3,bg=Dbg,fg=Dfg)
    
    settWlab1=tk.Label(settW,font=("나눔고딕",3))

    #위젯 배치 artiW
    settWla01.grid(row=0,column=0,sticky="nsew")
    settWlab1.grid(row=1,column=0,columnspan=2,sticky="nsew")
    settWla02.grid(row=2,column=0,sticky="nsew")
    settWen01.grid(row=2,column=1,sticky="nsew")
    settWla03.grid(row=3,column=0,sticky="nsew")
    settWen02.grid(row=3,column=1,sticky="nsew")
    settWen01.bind("<Return>", enter01)
    settWen02.bind("<Return>", enter02)

def W_miss():
    def on_select(event):
        global sname, scont, skt
        if(missWli01.curselection()[0]!=33):
            selected_index = missWli01.curselection() 
            #print(selected_index)
            sname = selected_index[0]
            scont = missile_cont[selected_index[0]]
            skt = missile_data[selected_index[0]][0]
            la03.config(text=name[sname])
            la05.config(text=scont)
            la07.config(text=skt)
            missWla04.config(text=name[sname])
            missWla06.config(text=scont)
            entext.set(skt)

        else:
            selected_index = 33    
            sname = 33
            scont = missile_cont[33]
            skt = 0
            la03.config(text="커스텀")
            la05.config(text=scont)
            la07.config(text=skt)
            missWla04.config(text="커스텀")
            missWla06.config(text=scont)
            entext.set(0)

    def enter01(event):
        global skt
        selected_index = 33    
        sname = 33
        scont = missile_cont[33]
        skt = float(eval(missWen01.get()))
        la03.config(text="커스텀")
        la05.config(text=scont)
        la07.config(text=skt)
        missWla04.config(text="커스텀")
        missWla06.config(text=scont)
    
    missW = tk.Toplevel(main)
    missW.title("핵 미사일 설정")
    missW.configure(bg="black")

    missWla01 = tk.Label(missW, text="핵 미사일 설정", font=("나눔고딕", 25, "bold"), bg=Dbg, fg=Dfg)
    missWla02 = tk.Label(missW, text="핵 미사일 선택", font=Dfont1, bg=Dbg, fg=Dfg, anchor="w")
    missWli01 = tk.Listbox(missW, bg=Dbg, fg=Dfg, selectmode='extended', font=Dfont3)
    for i in name:
        missWli01.insert(tk.END, i)
    missWlab1 = tk.Label(missW, font=("나눔고딕", 3))
    missWla03 = tk.Label(missW, text="핵 미사일 이름", font=Dfont3, bg=Dbg, fg=Dfg, anchor="w")
    missWla04 = tk.Label(missW, text=" ", font=Dfont3, bg=Dbg, fg=Dfg, anchor="e")
    missWla05 = tk.Label(missW, text="핵 미사일 국가", font=Dfont3, bg=Dbg, fg=Dfg, anchor="w")
    missWla06 = tk.Label(missW, text=" ", font=Dfont3, bg=Dbg, fg=Dfg, anchor="e")
    missWla07 = tk.Label(missW, text="핵 미사일 KT", font=Dfont3, bg=Dbg, fg=Dfg, anchor="w")
    entext=tk.StringVar()
    missWen01 = tk.Entry(missW, font=Dfont3, bg=Dbg, fg=Dfg, textvariable=entext)

    missWla01.grid(row=0, column=0, sticky="nsew")
    missWlab1.grid(row=1, column=0, columnspan=3, sticky="nsew")
    missWla02.grid(row=2, column=0, sticky="nsew")
    missWli01.grid(row=3, column=0, rowspan=10,sticky="nsew")
    missWla03.grid(row=3, column=1, sticky="nsew")
    missWla04.grid(row=3, column=2, sticky="nsew")
    missWla05.grid(row=4, column=1, sticky="nsew")
    missWla06.grid(row=4, column=2, sticky="nsew")
    missWla07.grid(row=5, column=1, sticky="nsew")
    missWen01.grid(row=5, column=2, sticky="nsew")
    missWli01.bind('<<ListboxSelect>>', on_select)
    missWen01.bind("<Return>", enter01)
    
def G_func():
    plt.close()
    x_test = np.linspace(0, 1, 100).reshape(-1, 1)
    y_pred = model.predict(x_test)

    # Inverse transform scaled predictions to original scale
    y_pred_original = scaler_y.inverse_transform(y_pred)

    labeel = [ '공기 폭발 반경(KM)', '화염구 반경(KM)', '강한 폭발 피해 반경(20psi/KM)',
              '보통 폭발 피해 반경(KM)', '열 복사 반경 (3도 화상/KM)', '약한 폭발 피해 반경 (1 psi/KM)', '방사선 반경 (500 rem/KM)']
    # Plot the 7 functions with different colors
    plt.figure(num='딥러닝 모델 함수',figsize=(10, 6))
    for i in range(7):
        plt.plot(x_test, y_pred_original[:, i], label=labeel[i])

    plt.xlabel('우라늄 양')
    plt.ylabel('피해')
    plt.title('딥러닝 모델 함수')
    plt.legend()
    plt.grid(True)
    plt.show()

def calc():
    global ar,fr,hr,mr,tr,lr,rr,dpeople,darea
    x_test = np.array([[skt]])
    x_test = scaler_x.transform(x_test)

    # 학습 데이터에 대한 예측값 생성.
    y_test_pred = model.predict(x_test)

    # 예측값의 스케일을 원래대로 되돌림.
    y_test_pred = scaler_y.inverse_transform(y_test_pred)

    # 각각의 변수에 저장
    ar = round(y_test_pred[0][0],5)
    fr = round(y_test_pred[0][1],5)
    hr = round(y_test_pred[0][2],5)
    mr = round(y_test_pred[0][3],5)
    tr = round(y_test_pred[0][4],5)
    lr = round(y_test_pred[0][5],5)
    rr = round(y_test_pred[0][6],5)
    '''
    # 결과 출력
    print("공기 폭발 넓이(KM):", ar)
    print("화염구 넓이(KM):", fr)
    print("강한 폭발 피해 넓이(20psi/KM):", hr)
    print("보통 폭발 피해 넓이(KM):", mr)
    print("열 복사 넓이 (3도 화상/KM):", tr)
    print("약한 폭발 피해 넓이 (1 psi/KM):", lr)
    print("방사선 넓이 (500 rem/KM):", rr)
    '''

    if lr*lr*3.14*densities>population:
        dpeople=population
    else : 
        dpeople=round(lr*lr*3.14*densities,0)
    
    if lr*lr*3.14>wide:
        darea=wide
    else:
        darea=round(lr*lr*3.14,5)

    la13.config(text=darea)
    la15.config(text=dpeople)

def W_dama():
    
    damaW=tk.Toplevel(main)
    damaW.title("피해 정보")
    damaW.configure(bg="black")

    damaWla01=tk.Label(damaW,text="피해 정보",font=("나눔고딕",25,"bold"),bg=Dbg,fg=Dfg,anchor="w")
    damaWla02=tk.Label(damaW,text="공기 폭발 반경(KM)",font=Dfont3,bg=Dbg,fg=Dfg,anchor="w")
    damaWla03=tk.Label(damaW,text=round(ar*ar*3.14,5),font=Dfont3,bg=Dbg,fg=Dfg,anchor="e")
    damaWla04=tk.Label(damaW,text="화염구 반경(KM)",font=Dfont3,bg=Dbg,fg=Dfg,anchor="w")
    damaWla05=tk.Label(damaW,text=round(fr*fr*3.14,5),font=Dfont3,bg=Dbg,fg=Dfg,anchor="e")
    damaWla06=tk.Label(damaW,text="강한 폭발 피해 반경(20psi/KM)",font=Dfont3,bg=Dbg,fg=Dfg,anchor="w")
    damaWla07=tk.Label(damaW,text=round(hr*hr*3.14,5),font=Dfont3,bg=Dbg,fg=Dfg,anchor="e")
    damaWla08=tk.Label(damaW,text="보통 폭발 피해 반경(KM)",font=Dfont3,bg=Dbg,fg=Dfg,anchor="w")
    damaWla09=tk.Label(damaW,text=round(mr*mr*3.14,5),font=Dfont3,bg=Dbg,fg=Dfg,anchor="e")
    damaWla10=tk.Label(damaW,text="열 복사 반경 (3도 화상/KM)",font=Dfont3,bg=Dbg,fg=Dfg,anchor="w")
    damaWla11=tk.Label(damaW,text=round(tr*tr*3.14,5),font=Dfont3,bg=Dbg,fg=Dfg,anchor="e")
    damaWla12=tk.Label(damaW,text="약한 폭발 피해 반경 (1 psi/KM)",font=Dfont3,bg=Dbg,fg=Dfg,anchor="w")
    damaWla13=tk.Label(damaW,text=round(lr*lr*3.14,5),font=Dfont3,bg=Dbg,fg=Dfg,anchor="e")
    damaWla14=tk.Label(damaW,text="방사선 반경 (500 rem/KM)",font=Dfont3,bg=Dbg,fg=Dfg,anchor="w")
    damaWla15=tk.Label(damaW,text=round(ar*ar*3.14,5),font=Dfont3,bg=Dbg,fg=Dfg,anchor="e")
    damaWla16=tk.Label(damaW,text="피해자",font=Dfont3,bg=Dbg,fg=Dfg,anchor="w")
    damaWla17=tk.Label(damaW,text=dpeople,font=Dfont3,bg=Dbg,fg=Dfg,anchor="e")
    damaWla18=tk.Label(damaW,text="피해 넓이",font=Dfont3,bg=Dbg,fg=Dfg,anchor="w")
    percentage = round((darea / wide) * 100,5)
    
    damaWla19 = tk.Label(damaW, text=f"{darea} | {percentage}% ", font=Dfont3, bg=Dbg, fg=Dfg, anchor="e")
    
    damaWlab1=tk.Label(damaW,font=("나눔고딕",3))
    damaWlab2=tk.Label(damaW,font=("나눔고딕",3))
    
    damaWla01.grid(row=0,column=0,sticky="nsew")
    damaWlab1.grid(row=1,column=0,columnspan=2,sticky="nsew")
    damaWla02.grid(row=2,column=0,sticky="nsew")
    damaWla03.grid(row=2,column=1,sticky="nsew")
    damaWla04.grid(row=3,column=0,sticky="nsew")
    damaWla05.grid(row=3,column=1,sticky="nsew")
    damaWla06.grid(row=4,column=0,sticky="nsew")
    damaWla07.grid(row=4,column=1,sticky="nsew")
    damaWla08.grid(row=5,column=0,sticky="nsew")
    damaWla09.grid(row=5,column=1,sticky="nsew")
    damaWla10.grid(row=6,column=0,sticky="nsew")
    damaWla11.grid(row=6,column=1,sticky="nsew")
    damaWla12.grid(row=7,column=0,sticky="nsew")
    damaWla13.grid(row=7,column=1,sticky="nsew")
    damaWla14.grid(row=8,column=0,sticky="nsew")
    damaWla15.grid(row=8,column=1,sticky="nsew")
    damaWlab2.grid(row=9,column=0,columnspan=2,sticky="nsew")
    damaWla16.grid(row=10,column=0,sticky="nsew")
    damaWla17.grid(row=10,column=1,sticky="nsew")
    damaWla18.grid(row=11,column=0,sticky="nsew")
    damaWla19.grid(row=11,column=1,sticky="nsew")

#위젯 선언 main
la01=tk.Label(main,text="NBS",font=("나눔고딕",50,"bold"),bg=Dbg,fg=Dfg)
bu01=tk.Button(main,text="핵 미사일 설정",font=Dfont1,bg=Dbg,fg=Dfg,anchor="e",command=W_miss)
la02=tk.Label(main,text="핵 미사일 이름",font=Dfont3,bg=Dbg,fg=Dfg,anchor="w")
la03=tk.Label(main,text=" ",font=Dfont3,bg=Dbg,fg=Dfg,anchor="e")
la04=tk.Label(main,text="핵 미사일 소유국",font=Dfont3,bg=Dbg,fg=Dfg,anchor="w")
la05=tk.Label(main,text=" ",font=Dfont3,bg=Dbg,fg=Dfg,anchor="e")
la06=tk.Label(main,text="핵 미사일 KT",font=Dfont3,bg=Dbg,fg=Dfg,anchor="w")
la07=tk.Label(main,text=" ",font=Dfont3,bg=Dbg,fg=Dfg,anchor="e")

bu02=tk.Button(main,text="발사 환경 설정",font=Dfont1,bg=Dbg,fg=Dfg,command=W_sett)
la08=tk.Label(main,text="넓이",font=Dfont3,bg=Dbg,fg=Dfg,anchor="w")
la09=tk.Label(main,text=wide,font=Dfont3,bg=Dbg,fg=Dfg,anchor="e")
la17=tk.Label(main,text="인구수",font=Dfont3,bg=Dbg,fg=Dfg,anchor="w")
la18=tk.Label(main,text=population,font=Dfont3,bg=Dbg,fg=Dfg,anchor="e")
la10=tk.Label(main,text="인구 밀도",font=Dfont3,bg=Dbg,fg=Dfg,anchor="w")
la11=tk.Label(main,text=densities,font=Dfont3,bg=Dbg,fg=Dfg,anchor="e")

bu03=tk.Button(main,text="피해 정보",font=Dfont1,bg=Dbg,fg=Dfg,command=W_dama)
la12=tk.Label(main,text="피해 넓이",font=Dfont3,bg=Dbg,fg=Dfg,anchor="w")
la13=tk.Label(main,text=" ",font=Dfont3,bg=Dbg,fg=Dfg,anchor="e")
la14=tk.Label(main,text="피해자",font=Dfont3,bg=Dbg,fg=Dfg,anchor="w")
la15=tk.Label(main,text=" ",font=Dfont3,bg=Dbg,fg=Dfg,anchor="e")

bu04=tk.Button(main,text="인공신경망 정보",font=Dfont2,bg=Dbg,fg=Dfg,command=W_arti)
bu05=tk.Button(main,text="프로그램 정보",font=Dfont2,bg=Dbg,fg=Dfg,command=W_info)

#la16=tk.Label(main,text="폭발 이미지")
img1=tk.PhotoImage(file="..\\image\\B1.png",master=button)
img2=tk.PhotoImage(file="..\\image\\B2.png",master=button)
bu07=tk.Button(button,image=img1,bg=Dbg,fg=Dfg,anchor="center", relief="flat",command=calc)

lab1=tk.Label(main,font=("나눔고딕",3))
lab2=tk.Label(main,font=("나눔고딕",3))
lab3=tk.Label(main,font=("나눔고딕",3))
lab4=tk.Label(main,font=("나눔고딕",3))

#위젯 배치 main
la01.grid(row=0,column=0)
lab1.grid(row=1,column=0,columnspan=3,sticky="nsew")
bu01.grid(row=2,column=0,rowspan=3,sticky="nsew")
la02.grid(row=2,column=1,sticky="nsew")
la03.grid(row=2,column=2,sticky="nsew")
la04.grid(row=3,column=1,sticky="nsew")
la05.grid(row=3,column=2,sticky="nsew")
la06.grid(row=4,column=1,sticky="nsew")
la07.grid(row=4,column=2,sticky="nsew")
lab2.grid(row=5,column=0,columnspan=3,sticky="nsew")
bu02.grid(row=6,column=0,rowspan=3,sticky="nsew")

la08.grid(row=6,column=1,sticky="nsew")
la09.grid(row=6,column=2,sticky="nsew")
la17.grid(row=7,column=1,sticky="nsew")
la18.grid(row=7,column=2,sticky="nsew")
la10.grid(row=8,column=1,sticky="nsew")
la11.grid(row=8,column=2,sticky="nsew")
lab3.grid(row=9,column=0,columnspan=3,sticky="nsew")
bu03.grid(row=10,column=0,rowspan=2,sticky="nsew")

la12.grid(row=10,column=1,sticky="nsew")
la13.grid(row=10,column=2,sticky="nsew")
la14.grid(row=11,column=1,sticky="nsew")
la15.grid(row=11,column=2,sticky="nsew")
lab4.grid(row=12,column=0,columnspan=3,sticky="nsew")
bu04.grid(row=13,column=0,sticky="nsew")
bu05.grid(row=13,column=1,sticky="nsew")

def on_button_click(event):
    bu07.config(image=img2)

def on_button_release(event):
    bu07.config(image=img1)
bu07.pack()
bu07.image = img1
bu07.bind("<Button-1>", on_button_click)
bu07.bind("<ButtonRelease-1>", on_button_release)

#메인루프 main
main.mainloop()
