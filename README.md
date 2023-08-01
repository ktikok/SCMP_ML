# SCMP_ML
ML studies in SCMP lab

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
    
    