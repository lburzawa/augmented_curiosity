import numpy as np
import matplotlib.pyplot as plt
import copy


A=np.loadtxt('./save1/test.txt',delimiter=',')
B=np.loadtxt('./save2/test.txt',delimiter=',')
#plt.ylim(ymin=-50)
plt.plot(A[:,1],A[:,3])
plt.plot(B[:,1],B[:,3])
#plt.plot(B[:,0],B[:,1])
plt.xlabel('Training steps')
plt.ylabel('Score')
plt.title('Comparing two different runs')
#plt.grid()
axes = plt.gca()
#axes.set_ylim([-60,60])
plt.savefig('./plot.png')

