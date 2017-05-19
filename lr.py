import sys
import re
import numpy as np
from math import log
from math import exp
from math import sqrt

MAX_ITERS = 100

# Load data from a file
def read_data(filename):
  f = open(filename, 'r')
  p = re.compile(',')
  data = []
  header = f.readline().strip()
  varnames = p.split(header)
  namehash = {}
  for l in f:
    example = [int(x) for x in p.split(l.strip())]
    x = example[0:-1]
    y = example[-1]
    data.append( (x,y) )
  return (data, varnames)

def gradient (w,x,y,b):
  act = np.dot(w,x)+ b
  denom = 1 + exp(y*act)
  grad = -(1/denom)*y
  return grad

def train_lr(data, eta, l2_reg_weight):
  numvars = len(data[0][0])
  w = [0.0] * numvars
  b = 0.0
  leng= len(data)
  #print (leng)
  weight_gradient=[0.0]*numvars
  
  for j in range(0,MAX_ITERS):
    grad1=0
    grad1=[0.0]* len(data)
    for i in range(0,numvars):
      w_gr = 0
      m=0
      
      
      for (x,y)in data:

        if (i==0):
          grad_int = (gradient(w,x,y,b))
          grad1[m] = grad_int
          w_gr =w_gr + (x[i]*grad_int)
        else:
            
          w_gr = w_gr + (x[i]*grad1[m])
        m+=1
      w_gr = w_gr + l2_reg_weight*w[i]
      weight_gradient[i]=w_gr
    bais_gradient =0
    bias_sum =0
    for (x,y) in data:
      bais_gradient = bais_gradient + gradient(w,x,y,b)
    bias_sum = bais_gradient
    sum_mag=0
    for i in range(0,numvars):
      sum_mag +=pow(weight_gradient[i],2)
    sum_mag +=pow(bias_sum,2)
    magnitude=sqrt(sum_mag)
    print (magnitude) 
    if magnitude < 0.00001:
      break
    
    for i in range(0,numvars):
      w[i]=w[i]-(eta*weight_gradient[i])
    b= b-(eta*bias_sum)
  return (w,b)


# Load train and test data.  Learn model.  Report accuracy.
def main(argv):
  if (len(argv) != 5):
    print ('Usage: lr.py <train> <test> <eta> <lambda> <model>')
    sys.exit(2)
  (train, varnames) = read_data(argv[0])
  (test, testvarnames) = read_data(argv[1])
  eta = float(argv[2])
  lam = float(argv[3])
  modelfile = argv[4]

  # Train model
  (w,b) = train_lr(train, eta, lam)

  # Write model file
  f = open(modelfile, "w+")
  f.write('%f\n' % b)
  for i in range(0,len(w)):
    f.write('%s %f\n' % (varnames[i], w[i]))

  # Make predictions, compute accuracy
  correct = 0
  for (x,y) in test:
    prob = 0.5
    act = -(np.dot(w,x)+b)
    deno =exp(act)
    prob = 1/deno
     
     # <-- YOUR CODE HERE
    #print (prob) 
    if (prob - 0.5) * y > 0:
      correct += 1
  acc = float(correct)/len(test)
  print ("Accuracy: ",acc)

if __name__ == "__main__":
  main(sys.argv[1:])