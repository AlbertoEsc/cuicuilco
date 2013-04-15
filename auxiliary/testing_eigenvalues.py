import numpy as np
from numpy import linalg as LA

if True:
    if True:
        Ax = [  [0.7, -0.5],
                        [-0.5, -0.1]]

        Ay = [  [-0.9, -0.5],
                        [-0.5, -0.1]]

        wx, vx = LA.eig(Ax)
        wxh, vxh = LA.eigh(Ax)

        print "First Test"
        print wx*vx
        print np.dot(Ax,vx)
                
        print "Test x0:"
        print wx[0]*vx[0]
        print np.dot(Ax, vx[0])
        print wx[0]*vx[:, 0]
        print np.dot(Ax, vx[:,0])
        
        print "Test xh0:"
        print wxh[0]*vxh[:,0]
        print np.dot(Ax, vxh[:,0])
        
        print ""
        print "Test x1:"
        print wx[1]*vx[:,1]
        print np.dot(Ax, vx[:,1])
        
        print "Test xh1:"
        print wxh[1]*vxh[:,1]
        print np.dot(Ax, vxh[:,1])
        wy, vy = LA.eig(Ay)
        wyh, vyh = LA.eigh(Ay)
        print ""
        
        print "Test y0:"
        print wy[0]*vy[:,0]
        print np.dot(Ay, vy[:,0])
        
        print "Test yh0:"
        print wyh[0]*vyh[0]
        print np.dot(Ay, vyh[0])
        print ""
        
        print "Test y1:"
        print wy[1]*vy[1]
        print np.dot(Ay, vy[1])
        print "Test yh1:"
        print wyh[1]*vyh[1]
        print np.dot(Ay, vyh[1])