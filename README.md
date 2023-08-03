# SCMP_ML
ML studies in SCMP lab

--------------------------------------
history
model = nn.Sequential(
    nn.Linear(2, 6),
    nn.ReLU(),
    nn.Linear(6, 3),
    nn.ReLU(),
    nn.Linear(3, 2),
    nn.ReLU(),
    nn.Linear(2, 1)
)
    #optimizer = optim.Adam(model.parameters(), lr=0.0001) # 10 loss:  47942025216.0
    #optimizer = optim.Adam(model.parameters(), lr=0.001) # 10 loss:   79399010304.0
    #optimizer = optim.Adam(model.parameters(), lr=0.00001) # 10 loss:439046144000.0
    optimizer = optim.Adam(model.parameters(), lr=0.0001)    #10 loss: 45356871680.0
    
    batch size = 10
    activation func = ReLU
    
---------------------------------------------
history
model = nn.Sequential(
    nn.Linear(2, 3),
    nn.ReLU(),
    nn.Linear(3, 2),
    nn.ReLU(),
    nn.Linear(2, 1)
)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)    #10 loss: 45356871680.0
    
    batch size = 100
    activation func = LeakyReLU
    
    learningRate: 3
    0 loss: 20956900.0
    1 loss: 14014879.0
    2 loss: 10895341.0
    3 loss: 8672112.0
    4 loss: 7091669.5
    5 loss: 5885813.0
    6 loss: 5454589.0
    7 loss: 4964886.5
    8 loss: 4256129.0
    9 loss: 3960994.25
    10 loss: 3603098.25
    MSE: 3603098.25
    RMSE: 1898.18
    
    learningRate: 0.05
    0 loss: 57799576.0
    1 loss: 5604386.0
    2 loss: 51211724.0
    3 loss: 38927144.0
    4 loss: 3202629.5
    5 loss: 1574894.25
    6 loss: 1391547.875
    7 loss: 1242121.25
    8 loss: 1263963.375
    9 loss: 2136653.0
    10 loss: 2127997.5
    MSE: 1242121.25
    RMSE: 1114.50
    
    earningRate: 0.025
    0 loss: 14092976.0
    1 loss: 1833870720.0
    2 loss: 17832964.0
    3 loss: 2018799744.0
    4 loss: 8180869.0
    5 loss: 4624224.5
    6 loss: 3856112.25
    7 loss: 2464013.5
    8 loss: 1026970.75
    9 loss: 236338.3125
    10 loss: 230703.3125
    MSE: 230703.31
    RMSE: 480.32
    
    learningRate: 0.03
    0 loss: 3281580032.0
    1 loss: 4815.013671875
    2 loss: 4433.93115234375
    3 loss: 3967.550048828125
    4 loss: 3500.9013671875
    5 loss: 3126.19775390625
    6 loss: 4127.388671875
    7 loss: 50248.15234375
    8 loss: 86064.1484375
    9 loss: 82545.1796875
    10 loss: 5908.564453125
    MSE: 3126.20
    RMSE: 55.91
    
    Bsize: 500 learningRate: 0.03
    10 loss: 100194.1171875
    MSE: 100194.12
    RMSE: 316.53
    
    Bsize: 500 learningRate: 0.03 RMSE: 111.31 1000 loss: 515.4253401318177
    
---------------------------------------------
model = nn.Sequential(
    nn.Linear(2, 5),
    nn.LeakyReLU(),
    nn.Linear(5, 3),
    nn.LeakyReLU(),
    nn.Linear(3, 1)
)

    for Bsize in [10, 50, 100, 500, 1000, 5000]:
    for learningRate in [ 1e-2, 5e-2, 1e-3, 5e-3, 1e-4, 5e-4, 1e-5, 5e-5, 1e-6, 5e-6, 1e-7, 5e-7,   ]:
    
    Bsize: 10 learningRate: 0.0005 minimum_RMSE: 147.55 epoch: 10 test_loss: 147.55229475511385 train_loss: 5.858115424511905
    
    Bsize: 100 learningRate: 0.05 minimum_RMSE: 81.86 epoch: 10 test_loss: 91.2877009018055 train_loss: 6.448340020483984
    Bsize: 500 learningRate: 0.01 minimum_RMSE: 58.02 epoch: 10 test_loss: 104.82666589571807 train_loss: 7.890926423311475

Bsize: 500 learningRate: 0.005 minimum_RMSE: 664.12 epoch: 10 test_loss: 664.1224708214593 train_loss: 146.62844332060885
Bsize: 500 learningRate: 0.05 minimum_RMSE: 595.02 epoch: 10 test_loss: 595.0220058955803 train_loss: 69.99861187239715
Bsize: 400 learningRate: 0.01 minimum_RMSE: 288.51 epoch: 10 test_loss: 288.510750449615 train_loss: 42.47055218306415
Bsize: 100 learningRate: 0.0005 minimum_RMSE: 597.13 epoch: 10 test_loss: 597.1273157794743 train_loss: 39.79032416883453
Bsize: 100 learningRate: 0.005 minimum_RMSE: 506.62 epoch: 10 test_loss: 15457.100892470102 train_loss: 525.0114284470386
Bsize: 100 learningRate: 0.001 minimum_RMSE: 679.17 epoch: 10 test_loss: 741.2088943071312 train_loss: 48.14275747873454
Bsize: 100 learningRate: 0.05 minimum_RMSE: 544.93 epoch: 10 test_loss: 840.0533465203266 train_loss: 82.63541207088521

Bsize: 400 learningRate: 0.01 minimum_RMSE: 523.19 epoch: 100 test_loss: 1512.7588208303398 train_loss: 329.1349694654155
Bsize: 100 learningRate: 0.001 minimum_RMSE: 468.47 epoch: 100 test_loss: 29206.6107585252 train_loss: 236.29854825252312
Bsize: 100 learningRate: 0.005 minimum_RMSE: 118.49 epoch: 100 test_loss: 315.69225768539206 train_loss: 12.983962510370366

Bsize: 100 learningRate: 0.005 minimum_RMSE: 39.03 epoch: 1000 test_loss: 90.99375407823331 train_loss: 8.282601415379995

epoch:  3350 test_loss:   889 train_loss:    10

n_of_data: 1000 Bsize: 10 learningRate: 0.001 minimum_RMSE: 906.87 epoch:  1000 test_loss:   918 train_loss:    38
n_of_data: 1000 Bsize: 10 learningRate: 0.005 minimum_RMSE: 809.15 epoch:  1000 test_loss:   946 train_loss:    30
n_of_data: 1000 Bsize: 10 learningRate: 0.0005 minimum_RMSE: 918.40 epoch:  1000 test_loss:   932 train_loss:    36
n_of_data: 1000 Bsize: 50 learningRate: 0.01 minimum_RMSE: 886.77 epoch:  1000 test_loss:   900 train_loss:   460
n_of_data: 1000 Bsize: 50 learningRate: 0.05 minimum_RMSE: 967.82 epoch:  1000 test_loss:   968 train_loss:   744
n_of_data: 1000 Bsize: 50 learningRate: 0.005 minimum_RMSE: 874.51 epoch:  1000 test_loss:   888 train_loss:   458
n_of_data: 1000 Bsize: 100 learningRate: 0.01 minimum_RMSE: 883.83 epoch:  1000 test_loss:   897 train_loss:   336
n_of_data: 1000 Bsize: 100 learningRate: 0.001 minimum_RMSE: 974.78 epoch:  1000 test_loss:   974 train_loss:   338
n_of_data: 1000 Bsize: 100 learningRate: 0.005 minimum_RMSE: 875.82 epoch:  1000 test_loss:   886 train_loss:   336
n_of_data: 1000 Bsize: 500 learningRate: 1e-07 minimum_RMSE: 994.67 epoch:  1000 test_loss:   994 train_loss:   731

NEW approach
-------------------------------
optimizer: Adadelta , n_of_data: 1000 , Bsize: 10 , learningRate: 0.1 , minimum_RMSE: 905.62 , epoch:  1000 , test_loss:   905 , train_loss:  
optimizer: Adadelta , n_of_data: 1000 , Bsize: 10 , learningRate: 0.1 , minimum_RMSE: 905.62 , epoch:  1000 , test_loss:   905 , train_loss:  
optimizer: Adadelta , n_of_data: 1000 , Bsize: 10 , learningRate: 0.05 , minimum_RMSE: 919.39 , epoch:  1000 , test_loss:   919 , train_loss: 
optimizer: Adagrad , n_of_data: 1000 , Bsize: 10 , learningRate: 0.1 , minimum_RMSE: 982.28 , epoch:  1000 , test_loss:   985 , train_loss:   
optimizer: AdamW , n_of_data: 1000 , Bsize: 10 , learningRate: 0.001 , minimum_RMSE: 862.54 , epoch:  1000 , test_loss:   943 , train_loss:   
optimizer: AdamW , n_of_data: 1000 , Bsize: 10 , learningRate: 0.005 , minimum_RMSE: 800.60 , epoch:  1000 , test_loss:   915 , train_loss:   
optimizer: AdamW , n_of_data: 1000 , Bsize: 10 , learningRate: 0.0005 , minimum_RMSE: 906.31 , epoch:  1000 , test_loss:   911 , train_loss:  
optimizer: AdamW , n_of_data: 1000 , Bsize: 100 , learningRate: 0.001 , minimum_RMSE: 979.86 , epoch:  1000 , test_loss:   980 , train_loss:  
optimizer: Adamax , n_of_data: 1000 , Bsize: 10 , learningRate: 0.001 , minimum_RMSE: 855.71 , epoch:  1000 , test_loss:   978 , train_loss:    40 , layer_dim_list: [2, 16, 8, 1]
optimizer: Adamax , n_of_data: 1000 , Bsize: 10 , learningRate: 0.005 , minimum_RMSE: 826.29 , epoch:  1000 , test_loss:   958 , train_loss:    40 , layer_dim_list: [2, 16, 8, 1]
optimizer: Adamax , n_of_data: 1000 , Bsize: 10 , learningRate: 0.0001 , minimum_RMSE: 825.55 , epoch:  1000 , test_loss:   998 , train_loss:    44 , layer_dim_list: [2, 16, 8, 1]
optimizer: Adamax , n_of_data: 1000 , Bsize: 10 , learningRate: 0.0005 , minimum_RMSE: 823.91 , epoch:  1000 , test_loss:   989 , train_loss:    44 , layer_dim_list: [2, 16, 8, 1]
optimizer: Adamax , n_of_data: 1000 , Bsize: 10 , learningRate: 1e-05 , minimum_RMSE: 823.85 , epoch:  1000 , test_loss:   846 , train_loss:    29 , layer_dim_list: [2, 16, 8, 1]
optimizer: Adamax , n_of_data: 1000 , Bsize: 10 , learningRate: 5e-05 , minimum_RMSE: 823.67 , epoch:  1000 , test_loss:   985 , train_loss:    43 , layer_dim_list: [2, 16, 8, 1]
optimizer: Adamax , n_of_data: 1000 , Bsize: 10 , learningRate: 1e-06 , minimum_RMSE: 823.64 , epoch:  1000 , test_loss:   823 , train_loss:    26 , layer_dim_list: [2, 16, 8, 1]
optimizer: Adamax , n_of_data: 1000 , Bsize: 10 , learningRate: 5e-06 , minimum_RMSE: 823.62 , epoch:  1000 , test_loss:   830 , train_loss:    28 , layer_dim_list: [2, 16, 8, 1]
optimizer: Adamax , n_of_data: 1000 , Bsize: 10 , learningRate: 1e-07 , minimum_RMSE: 823.62 , epoch:  1000 , test_loss:   823 , train_loss:    26 , layer_dim_list: [2, 16, 8, 1]
optimizer: Adamax , n_of_data: 1000 , Bsize: 10 , learningRate: 5e-07 , minimum_RMSE: 823.58 , epoch:  1000 , test_loss:   823 , train_loss:    26 , layer_dim_list: [2, 16, 8, 1]
optimizer: Adamax , n_of_data: 1000 , Bsize: 50 , learningRate: 0.05 , minimum_RMSE: 729.09 , epoch:  1000 , test_loss:   900 , train_loss:   481 , layer_dim_list: [2, 16, 8, 1]
optimizer: Adamax , n_of_data: 1000 , Bsize: 50 , learningRate: 0.001 , minimum_RMSE: 724.83 , epoch:  1000 , test_loss:   912 , train_loss:   463 , layer_dim_list: [2, 16, 8, 1]
optimizer: Adamax , n_of_data: 1000 , Bsize: 50 , learningRate: 0.005 , minimum_RMSE: 717.31 , epoch:  1000 , test_loss:   935 , train_loss:   457 , layer_dim_list: [2, 16, 8, 1]
optimizer: Adamax , n_of_data: 1000 , Bsize: 50 , learningRate: 0.0001 , minimum_RMSE: 717.26 , epoch:  1000 , test_loss:   883 , train_loss:   477 , layer_dim_list: [2, 16, 8, 1]
optimizer: Adamax , n_of_data: 1000 , Bsize: 50 , learningRate: 0.0005 , minimum_RMSE: 717.12 , epoch:  1000 , test_loss:   907 , train_loss:   460 , layer_dim_list: [2, 16, 8, 1]
optimizer: Adamax , n_of_data: 1000 , Bsize: 50 , learningRate: 1e-05 , minimum_RMSE: 717.12 , epoch:  1000 , test_loss:   724 , train_loss:   760 , layer_dim_list: [2, 16, 8, 1]
optimizer: Adamax , n_of_data: 1000 , Bsize: 50 , learningRate: 5e-05 , minimum_RMSE: 717.10 , epoch:  1000 , test_loss:   850 , train_loss:   519 , layer_dim_list: [2, 16, 8, 1]
optimizer: Adamax , n_of_data: 1000 , Bsize: 50 , learningRate: 1e-06 , minimum_RMSE: 717.10 , epoch:  1000 , test_loss:   717 , train_loss:   829 , layer_dim_list: [2, 16, 8, 1]
optimizer: Adamax , n_of_data: 1000 , Bsize: 50 , learningRate: 5e-06 , minimum_RMSE: 717.09 , epoch:  1000 , test_loss:   719 , train_loss:   798 , layer_dim_list: [2, 16, 8, 1]
optimizer: Adamax , n_of_data: 1000 , Bsize: 50 , learningRate: 1e-07 , minimum_RMSE: 717.09 , epoch:  1000 , test_loss:   717 , train_loss:   837 , layer_dim_list: [2, 16, 8, 1]
optimizer: Adamax , n_of_data: 1000 , Bsize: 50 , learningRate: 5e-07 , minimum_RMSE: 717.09 , epoch:  1000 , test_loss:   717 , train_loss:   833 , layer_dim_list: [2, 16, 8, 1]
optimizer: Adamax , n_of_data: 1000 , Bsize: 100 , learningRate: 0.1 , minimum_RMSE: 863.93 , epoch:  1000 , test_loss:   879 , train_loss:   345 , layer_dim_list: [2, 16, 8, 1]
optimizer: RMSprop , n_of_data: 1000 , Bsize: 10 , learningRate: 5e-05 , minimum_RMSE: 945.54 , epoch:  1000 , test_loss:   977 , train_loss:    28 , layer_dim_list: [2, 16, 8, 1]
optimizer: RMSprop , n_of_data: 1000 , Bsize: 10 , learningRate: 1e-06 , minimum_RMSE: 945.30 , epoch:  1000 , test_loss:   952 , train_loss:    28 , layer_dim_list: [2, 16, 8, 1]
optimizer: RMSprop , n_of_data: 1000 , Bsize: 10 , learningRate: 1e-07 , minimum_RMSE: 944.07 , epoch:  1000 , test_loss:   945 , train_loss:    27 , layer_dim_list: [2, 16, 8, 1]
