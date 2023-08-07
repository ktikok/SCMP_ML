from datetime import date
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from torchvision import datasets, transforms

import copy
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import plotly.express as px
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm

def save_my_work(model, logfile, xdata_name, opt, N, Bsize, learningRate, epoch, second_time, best_mse, mse, loss, init_time):
    model_save_time = str(int(time.time()))
    model_file_name = '/content/drive/MyDrive/0_318lab/SCMP_ML/'+'model' + model_save_time + '.pt'
    torch.save(model, model_file_name )
    print(
                    datetime.now(),
                    'data:', xdata_name,
                    file=logfile
    )
    print(
                    ', layer_dim_list:', dim_list,
                    'activateion:', act,
                    ', optimizer:', opt,
                    ', n_of_data:',N,
                    ', Bsize:', Bsize ,
                    ', learningRate:', learningRate ,
                    file=logfile
                    )
    print(
                ', epoch: %5d' % epoch,
                ', passed_time: %.3f' % ( ( ( time.time()-second_time ) )  / 60  ),  'm',

                ", minimum_RMSE: %.2f" % (best_mse),
                ', test_loss: %2f' % (mse),
                ', train_loss: %2f' % (loss),
                ', passed_time_accum: %.3f' % ( ( init_time - time.time() )  / 60  ),  'm',
                file=logfile
                )
    plt.plot(history)
    plt.yscale('log')
    plt.title('test_loss')
    plt.savefig(
                model_file_name[0:-3] + 'test_loss' + '.pdf',
                format="pdf",
                bbox_inches="tight"
                )
    plt.show()

    plt.plot(history_train)
    plt.title('train_loss')
    plt.yscale('log')
    plt.savefig(
                model_file_name[0:-3] + 'train_loss' + '.pdf',
                format="pdf",
                bbox_inches="tight"
                )
    plt.show()
    
    
def print_progress(epoch, mse, loss, second_time):
    print('epoch: %5d' % epoch,
        ', test_loss: {:7.1f}'.format(mse),
        ', train_loss: {:7.1f}'.format(loss),
        ', est_time: {:5.1f}'.format(( ( ( time.time()-second_time ) )  / 60  )),  'min,',
        'average_time: {:.2f}'.format( ( time.time()-second_time )/(epoch+1)), 's'
        )
    
N, D_in, D_out = 10000, 2, 1

xdata_name = 'data/KEdataX_N_' + str(N) + '_Interval_10_1691050560.pt'
ydata_name = 'data/KEdataY_N_' + str(N) + '_Interval_10_1691050560.pt'
X = torch.load(xdata_name)
y = torch.load(ydata_name)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, shuffle=False)


#default

"""
try

activateion: Softplus , optimizer: Adam , n_of_data: 10000 , Bsize: 100 , learningRate: 0.0005 , minimum_RMSE: 27.47 , epoch:  3660 , test_loss: 595.456665 , train_loss: 14.215258 , layer_dim_list: [2, 20, 20, 20, 20, 1] , passed_time: 5.000 m , passed_time: -160.696 s

activateion: Softplus , optimizer: Adam , n_of_data: 10000 , Bsize: 100 , learningRate: 5e-05 , minimum_RMSE: 252.92 , epoch:  3668 , test_loss: 256.572418 , train_loss: 184.996445 , layer_dim_list: [2, 20, 20, 20, 20, 1] , passed_time: 5.000 m , passed_time: -170.736 s

activateion: Softplus , optimizer: Adam , n_of_data: 10000 , Bsize: 10 , learningRate: 0.005 , minimum_RMSE: 77.75 , epoch:   386 , test_loss: 3821.791748 , train_loss: 5.220962 , layer_dim_list: [2, 20, 20, 20, 20, 1] , passed_time: 5.010 m , passed_time: -30.134 s

activateion: Softplus , optimizer: Adam , n_of_data: 10000 , Bsize: 500 , learningRate: 0.001 , minimum_RMSE: 22.67 , epoch: 13813 , test_loss: 15479.684570 , train_loss: 1186.494629 , layer_dim_list: [2, 20, 20, 20, 20, 1] , passed_time: 5.000 m , passed_time: -205.868 s

"""
# for learningRate in range(10): #


dim_list = [2, 20, 20, 20, 20, 1]


f = open("/content/drive/MyDrive/0_318lab/SCMP_ML/log.txt", "a")
#If data is less complex and is having fewer dimensions or features then neural networks with 1 to 2 hidden layers would work.
# If data is having large dimensions or features then to get an optimum solution, 3 to 5 hidden layers can be used.

init_time = time.time()
print('start')
for act in [
            # 'LeakyReLU',
            # 'LogSigmoid',
            'Softplus']:

    for opt in [
                # 'Adadelta',
                # 'Adagrad',
                'Adam',
                # 'AdamW',
                # 'SparseAdam',
                # 'Adamax',
                # 'ASGD',
                # 'LBFGS',
                # 'NAdam',
                # 'RAdam',
                # 'RMSprop',
                # 'Rprop',
                # 'SGD'
                ]:
        try:
            # for Bsize in [ 10, 50, 100, 500]:
            # for Bsize in [ 10, 100, 1000]:
            for batch_size in [ 10]:

                # for learningRate in [ 1e-1, 1e-2, 1e-3, 1e-4,1e-5, 1e-6, ]:
                # for learningRate in [ 1e-1, 5e-1, 1e-2, 5e-2, 1e-3, 5e-3, 1e-4, 5e-4, 1e-5, 5e-5, 1e-6, 5e-6, ]:
                for learningRate in [  0.005 ]:

                    make_model = 'model = nn.Sequential('
                    for layer_num in range( len(dim_list) - 1 ):
                        if layer_num == len(dim_list) - 2:
                            make_model = make_model + 'nn.Linear(dim_list[' + str(layer_num) + '], dim_list[' + str(layer_num+1) + ']) )'
                        else:
                            # make_model = make_model + 'nn.Linear(dim_list[' + str(layer_num) + '], dim_list[' + str(layer_num+1) + ']), nn.LeakyReLU(),'
                            make_model = make_model + 'nn.Linear(dim_list[' + str(layer_num) + '], dim_list[' + str(layer_num+1) + ']), nn.' + act + '(),'


                    exec(make_model)

                    exec('optimizer = optim.' + opt + '(model.parameters(), lr=learningRate  )')

                    batch_start = torch.arange(0, len(X_train), batch_size)

                    # Hold the best model
                    best_mse = np.inf   # init to infinity
                    best_weights = None
                    history = []
                    history_train = []


                    # for epoch in range(n_epochs):
                    second_time = time.time()

                    epoch = 0
                    len_batch = len(X_train)
                    while(1):

                        model.train()

                        for start in range(0,len_batch, batch_size):

                            X_batch = X_train[start:start+batch_size] # 두번째 값이 길이를 초과해도 오류안뜨고 그냥 시작부터 끝까지 출력해주는 착한친구.
                            y_batch = y_train[start:start+batch_size]
                            # forward pass
                            y_pred = model(X_batch).squeeze()

                            before_loss = torch.mean( (y_pred/y_batch-1)**2 + (y_batch/y_pred-1)**2 + (y_pred-y_batch)**2 )

                            loss = before_loss**(1/2)

                            # backward pass
                            optimizer.zero_grad()
                            loss.backward()

                            # update weights
                            optimizer.step()

                        loss = float(loss)

                        history_train.append(loss)

                        model.eval()
                        y_pred = model(X_test).squeeze()
                        #print(y_pred)

                        before_loss = torch.mean(  (y_pred/y_test-1)**2 + (y_test/y_pred-1)**2 + (y_pred-y_test)**2 )

                        mse = before_loss**(1/2)

                        mse = float(mse)

                        history.append(mse)
                        if mse < best_mse:
                            best_mse = mse
                            best_weights = copy.deepcopy(model.state_dict())


                        if epoch % 100 == 0:
                            # print('epoch: %5d' % epoch, 'test_loss: %.2f' % (mse), 'train_loss: %.2f' % (loss), 'est_time: %.2f' % (( ( epoch_time ) / (epoch+1) ) / 60 * (n_epochs-epoch) ) ,'min', "epoch_time:", epoch_time, 's')
                            print_progress(epoch, mse, loss, second_time)
                            # print('{:.6}'.format(val))
                            # print("{:10.4f}".format(x))

                        # if (mse<10) or (str(mse)=='nan') or ( ( ( time.time()-second_time ) )  / 60 > 0.1  ):
                        if (mse<10) or (str(mse)=='nan'):
                            print_progress(epoch, mse, loss, second_time)
                            break
                        epoch = epoch + 1

                    # restore model and return best accuracy
                    model.load_state_dict(best_weights)
                    # print("MSE: %.2f" % best_mse)
                    # print("RMSE: %.2f" % np.sqrt(best_mse))
                    save_my_work(model, f, xdata_name, opt, N, batch_size, learningRate, epoch, second_time, best_mse, mse, loss, init_time)

                    print('finish--------------------------------------------------------------------------------------------------------------------------------------------------------')
                    print('')

        except:
            # os.mkdir('/content/drive/MyDrive/0_318lab/SCMP_ML/')
            save_my_work(model, f, xdata_name, opt, N, batch_size, learningRate, epoch, second_time, best_mse, mse, loss, init_time)

f.close()
print('all done')
# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------