'''
    draw picture of test data and true data
    author:Cuson
    date:2019/1/18
'''

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def drawPic():
    # x=  np.linspace(0,205,206)
    test_y = pd.read_csv('test4.csv').drop(labels=['id'],axis=1)
    true_y= pd.read_csv('testY.csv').drop(labels=['id'],axis=1)


    # print(test_y)
    # return test_y,true_y
    plt.plot(test_y,color='blue',label='predict')
    plt.plot(true_y,color='green',label='true')
    plt.show()


if __name__ =='__main__':
    drawPic()
