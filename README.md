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

    