import numpy as np
np.set_printoptions(precision=2)

strategy = 'final'

print('distance')
print(np.loadtxt(f'./data/dist.txt').mean(), np.loadtxt(f'./data/dist.txt').std())
print()

print('goto')
print(np.loadtxt(f'./data/goto.txt').mean(), np.loadtxt(f'./data/goto.txt').std(), np.loadtxt(f'./data/goto.txt').mean()/50000)
print()

print('recharging')
print(np.loadtxt(f'./data/recharging.txt').mean(), np.loadtxt(f'./data/recharging.txt').std(), np.loadtxt(f'./data/recharging.txt').mean()/50000)
print()

print('wait')
print(np.loadtxt(f'./data/wait.txt').mean(), np.loadtxt(f'./data/wait.txt').std(), np.loadtxt(f'./data/wait.txt').mean()/50000)
print()

print('treasure')
print(np.loadtxt(f'./data/treasure.txt').mean(), np.loadtxt(f'./data/treasure.txt').std())
