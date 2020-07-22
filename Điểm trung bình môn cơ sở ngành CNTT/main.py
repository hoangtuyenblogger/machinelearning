import csv
import pandas as pd
import statistics as st
import numpy as np
import random
import math
import sys
import matplotlib.pyplot as plt
import Pearson_Spearman as PS

def main():
    filename = 'data.csv'
    data = pd.read_csv(filename)

    ## load dữ liệu vào mảng
    SV = data["SV"]
    NMCNTT = data["NMCNTT"]
    CSLT = data["CSLT"]
    KTLT = data["KTLT"]
    CTDLGT = data["CTDLGT"]
    MMT = data["MMT"]

    print('Đối tượng thực hiện khảo sát chủ yếu là sinh viên năm', st.mode(SV))
    ####################################
    print('################Tính điểm nhỏ nhất trong từng môn học#########################')
    print('Điểm nhỏ nhất NMCNTT là: ',NMCNTT.min())
    print('Điểm nhỏ nhất CSLT là: ', CSLT.min())
    print('Điểm nhỏ nhất KTLT là: ', KTLT.min())
    print('Điểm nhỏ nhất CTDLGT là: ', CTDLGT.min())
    print('Điểm nhỏ nhất MMT là: ', MMT.min())
    print('################Tính điểm lớn nhất trong từng môn học#########################')
    print('Điểm lớn nhất NMCNTT là: ',NMCNTT.max())
    print('Điểm lớn nhất CSLT là: ', CSLT.max())
    print('Điểm lớn nhất KTLT là: ', KTLT.max())
    print('Điểm lớn nhất CTDLGT là: ', CTDLGT.max())
    print('Điểm lớn nhất MMT là: ', MMT.max())
    #################################################################
    print('################Tính điểm trung bình trong từng môn học#########################')
    ndigit = 1
    print('Điểm trung bình đối với từng môn học là:')
    print('Mean_NMCNTT= {0} \nMean_CSLT= {1} \nMean_KTLT= {2} \nMean_CTDLGT= {3} \nMean_MMT={4}'.format(round(NMCNTT.mean(),ndigit), round(CSLT.mean(),ndigit), round(KTLT.mean(),ndigit), round(CTDLGT.mean(),ndigit), round(MMT.mean(),ndigit)))
    print('#########################################')
    print('Điểm xuất hiện nhiều nhất của NMCNTT là : ',st.mode(NMCNTT))
    print('Điểm xuất hiện nhiều nhất của CSLT là : ', st.mode(CSLT))
    print('Điểm xuất hiện nhiều nhất của KTLT là : ', st.mode(KTLT))
    print('Điểm xuất hiện nhiều nhất của CTDLGT là : ', st.mode(CTDLGT))
    print('Điểm xuất hiện nhiều nhất của MMT là : ', st.mode(MMT))
    print('####Phân tích độ tương quan giữa các đại lượng  Pearson  và Spearman#####')
    
    print('Spearon giữa CSLT và CTDLGT là: ', PS.Pearson(CSLT, CTDLGT))
    print('Spearon giữa KTLT và CTDLGT là: ',PS.Pearson(KTLT,CTDLGT))
    print('#########################################')
    #biểu đồ Spearman
    plt.plot(KTLT,CTDLGT)
    plt.axis([1, 10, 1, 10])
    plt.xlabel('KTLT')
    plt.ylabel('CTDLGT')
    plt.show()
if __name__ == '__main__':
    main()


