import sys
sys.path.append("/home/escalafl/usr/lib/python2.6/site-packages")
import mdp
from mdp import numx

def identity(x): return x

def u3(x): return numx.absolute(x)**3 #A simple nonlinear transformation

def max_feature(x): #Computes the maximum feature within each sample
    return x.max(axis=1).reshape((-1,1)) 
  
x = numx.array([[-2., 2.], [0.2, 0.3], [1.2, 0.6]])
gen = mdp.nodes.GeneralExpansionNode(funcs=[identity, u3, max_feature])
print(gen.execute(x))

quit()

def u08(x): return numx.abs(x)**0.8 #A simple nonlinear transformation

def norm2(x): #Computes the norm of each sample as an Nx1 column array
    return ((x**2).sum(axis=1)**0.5).reshape((-1,1)) 
    
def second_degree_monomials(x):
    dim = x.shape[1]

    dexp = numx.zeros((dim*(dim+1)/2, x.shape[0]))
    prec = x.T
    k = 0
    for j in range(dim):
        factor = prec[j:, :]
        len_ = factor.shape[0]
        #print factor.shape, x.shape, dexp.shape, dim, j
        dexp[k:k+len_, :] = x[:, j] * factor
        k = k+len_
    return dexp.T

x0 = numx.linspace(-1.704, 1.704, 30) #Slow parameter, not directly included in the data x

x = numx.random.normal(size=(30,4))
x1 = x[:,0]
x[:,1] = x1 * x0
x[:,2] = 0.1*(x1**2) + 0.2*x0*x1 + 0.4*x0*(x1**2) + 0.8*(x0 * x1)**2+0.15*x0

func1 = [identity]
func2 = [identity, u08]
func3 = [identity, second_degree_monomials]

print "Original data", x[:,2]
print "Original slow parameter:", x0

for func in [func1, func2, func3]:
    GEN_node = mdp.nodes.GeneralExpansionNode(func)
    SFA_node = mdp.nodes.SFANode()
    flow = mdp.Flow([GEN_node, SFA_node])
    flow.train(x)
    y = flow.execute(x)
    print "Slowest feature found by SFA:", y[:,0]
    print "has delta value:", SFA_node.d[0]
quit()

def multiply(x1, x2):
    return x1 * x2

def products_2(x, func):
    x_height, x_width = x.shape

    k=0
    mask = numx.triu(numx.ones((x_width,x_width)), k) > 0.5

    z1 = x.reshape(x_height, x_width, 1)
    z2 = x.reshape(x_height, 1, x_width)
    yexp = func(z1, z2) # twice computation, but performance gain due to lack of loops

    out = yexp[:, mask]
    return out 

x = numx.random.normal(size=(100,800))
y1 = second_degree_monomials(x)

#y2 = products_2(x, multiply)
#print numx.allclose(y1,y2)
print "ok"

