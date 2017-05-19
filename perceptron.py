import sys
import re
from math import log
from math import exp

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
    #print(y)
    # Each example is a tuple containing both x (vector) and y (int)
    data.append( (x,y) )
  return (data, varnames)


# Learn weights using the perceptron algorithm
def train_perceptron(data):
    # Initialize weight vector and bias
    #x = data[0];
    #print(x);
    #return

    numvars = len(data[0][0])
    #print (numvars)
    w = [0.0] * numvars
    b = 0.0
    #print (b)
    a = [0.0] * numvars
    act=0
    #correct =0
    rows = len(data)

    #print (rows)
    for iter in range(0,MAX_ITERS):
      #b=0.0
      correct=0
      for d in data:
        #for row in range(0,rows):
      #print (b)
        (x,y)= d
        act = 0
        for i in range(0,numvars):
          act = act + (w[i]*x[i])
        act= act + b
        #print (act)
        if (y*act <= 0.0):
          for i in range(0,numvars):
            w[i] = w[i] + (y*x[i])
          b = y + b
        #print (b)
        else:
         correct+=1
      #print(correct)
        if(correct==rows):
          break
    #print (len(w))
    


    #
    # YOUR CODE HERE!
    #

    return (w,b)


# Load train and test data.  Learn model.  Report accuracy.
def main(argv):
  # Process command line arguments.
  # (You shouldn't need to change this.)
  if (len(argv) != 3):
    print ('Usage: perceptron.py <train> <test> <model>')
    sys.exit(2)
  (train, varnames) = read_data(argv[0])
  (test, testvarnames) = read_data(argv[1])
  modelfile = argv[2]

  # Train model
  (w,b) = train_perceptron(train)
  #print (b)
  #print(w,b)
  # Write model file
  # (You shouldn't need to change this.)
  f = open(modelfile, "w+")
  f.write('%f\n' % b)
  for i in range(len(w)):
    f.write('%s %f\n' % (varnames[i], w[i]))

  # Make predictions, compute accuracy
  correct = 0
  numvars = len(w)
  a = [0.0] * numvars
  sum = 0
  activation=0
  for exp in test:
    activation = 0.0
    #X = [x]
    for i in range(0,numvars):
      #print (w[i],x[i])
      #print (x[i])
      #print (sum)
      #print ("////")
      activation = activation + (w[i]*exp[0][i])
    activation =activation+b
         # <-- YOUR CODE HERE
    #print ("activation",activation)
    if activation * exp[1] > 0:
      correct += 1
  

  acc = float(correct)/len(test)
  print ("Accuracy: ",acc)

if __name__ == "__main__":
  main(sys.argv[1:])