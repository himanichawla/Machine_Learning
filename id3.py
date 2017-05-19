import sys
import re
import math
import node
def read_data(filename):
  f = open(filename, 'r')
  p = re.compile(',')
  data = []
  header = f.readline().strip()
  varnames = p.split(header)
  namehash = {}
  for l in f:
    data.append([int(x) for x in p.split(l.strip())])
  return (data, varnames)

def entropy(data, varnames):
  '''
  print '\n@@@@@@@@@@'
  print data
  print varnames
  print '@@@@@@@@@@@@@\n'
'''
  if not data:
    '''
    print '\n1111111111111'
    print data
    print varnames
    print '@@@@@@@@@@@@@\n'
    '''
    return 0
  length = float(len(data))
  pos = 0
  neg = 0
  #for i in range(0,len(varnames)):
  for row in range(0,len(data)):
    if data[row][len(varnames)-1]==1:
      pos+=1
    else:
      neg+=1
  if(pos==neg):
    entropyTotal=1
  elif(pos==0 or neg==0):
    '''
    print '\n222222222222222'
    print data
    print varnames
    print '@@@@@@@@@@@@@\n'
    '''
    entropyTotal=0
  elif(length==0):
    '''
    print '\n3333333333333333'
    print data
    print varnames
    print '@@@@@@@@@@@@@\n'
    '''
    entropyTotal=0
  else:
   ptrue = pos/length
   pfalse = neg/length
   entropyT = math.log(ptrue if ptrue>0 else 1,2)
   entropyF = math.log(pfalse if pfalse>0 else 1,2)
   entropyTotal =-(ptrue*entropyT + pfalse*entropyF)
  #print 'entro pos', pos
  #print 'entro neg', neg
  return entropyTotal

def gain (data,varnames,attr):

  length = float(len(data))
  numT = 0
  numF = 0
  trueEx = []
  falseEx = []
  index = varnames.index(attr)
  #print index
  #for i in range(0,len(varnames1)):
  for row in range(0,len(data)):
    if data[row][index]==1:
       numT += 1
       trueEx.append(data[row])
    elif data[row][index] == 0:
       numF += 1
       falseEx.append(data[row])
  ptrue = numT/length
  pfalse = numF/length
  #result1 =ptrue* entropy(trueEx,varnames)
  #result2= pfalse*entropy(falseEx,varnames)
  #result3= entropy(data,varnames)
  #print data
  result = entropy(data, varnames) - ptrue*entropy(trueEx, varnames) - pfalse*entropy(falseEx, varnames)
  #print result1, result2,result3
  return result

def positive_ex(data,varnames,attr):
  length = float(len(data))
  numT = 0
  trueEx = []
  index = varnames.index(attr)
  #print index
  #for i in range(0,len(varnames1)):
  for row in range(0,len(data)):
    if data[row][index]==1:
       numT += 1
       trueEx.append(data[row])
  return trueEx

def negative_ex(data,varnames,attr):
  length = float(len(data))
  numF = 0
  falseEx = []
  index = varnames.index(attr)
  #print index
  #for i in range(0,len(varnames1)):
  for row in range(0,len(data)):
    if data[row][index]==0:
       numF += 1
       falseEx.append(data[row])
  return falseEx



def chooseAttribute(train,varnames):

    best = varnames[0]
    maxGain = 0;
    for i in range(0,len(varnames)-1):
        newGain = gain( train, varnames, varnames[i])
         
        if newGain>maxGain:
            maxGain = newGain
            best = varnames[i]
    return best


def print_model(root, modelfile):
  f = open(modelfile, 'w+')
  root.write(f, 0)

def guess(train,varnames):
  l=0
  m=0
  for row in train:
    if(row[len(varnames)-1]==1):
      l+=1
    else:
      m+=1
  if(l>m):
    return 1
  else:
    return 0





def build_tree(train, varnames):
     '''
     print '\n##########'
     print varnames
     print train
     print '###########\n'
    '''
     common = guess(train,varnames)

     class_0 = [ row[-1] for row in train if row[-1] == 0]
     #class_1 = [ row[-1] for row in train if row[-1] == 1]

     #print class_0
     #print class_1

     if (len(class_0) == len(train)): # Homogeneous
        return node.Leaf("abc",0)
     elif (len(class_0) == 0):
        return node.Leaf("abc",1)

     if (len(varnames) == 0):
        return node.Leaf(varnames,common)

     else:

        best = chooseAttribute(train,varnames)
        best_gain = gain(train,varnames,best)
        thres_gain =0.0
        #print best_gain
        #print best
        Pos = positive_ex(train,varnames,best)
        Neg = negative_ex(train,varnames,best)
        i=varnames.index(best)
        #print best, i
        varname_copy = varnames[:]
        #varname_copy.remove(best)
     
     #n1=len(Neg)
 
     
        if(best_gain==thres_gain):
           return node.Leaf(best,common)
        elif(best_gain>thres_gain):
           root = node.Split(varname_copy,i,build_tree(Neg,varname_copy),build_tree(Pos,varname_copy))
           return root
        #return root

      #root=node.split(varnames,best,Pos,Neg)
      #print root


    # >>>> YOUR CODE GOES HERE <<<<
    # For now, always return a leaf predicting "1":
     #return node.Leaf(varnames, 1)


def main(argv):
    if (len(argv) != 3):
     print ('Usage: id3.py <train> <test> <model>')
     sys.exit(2)

    (train, varnames) = read_data(argv[0])
    (test, testvarnames) = read_data(argv[1])
    modelfile = argv[2]
    n=len(varnames)
    #print n
    #attributes=varnames[:n-1]
    #attribute = varnames[1]
    #chooseAttribute(train,varnames)
    #gain(train,varnames,attribute)
    #entropy(train,varnames)
    #build_tree(train,varnames)
    root = build_tree(train,varnames)
    #print root
    #root = build_tree(train, varnames)
    root.write(sys.stdout, 0)


    print_model(root, modelfile)
    correct = 0
  # The position of the class label is the last element in the list.
    yi = len(test[0]) - 1
    for x in test:
      pred = root.classify(x)
      if pred == x[yi]:
        correct += 1
        #print 'Correct: ', x

    acc = float(correct)/len(test)
    print ("Accuracy: ",acc)

    #print root

if __name__ == "__main__":
  main(sys.argv[1:])

  


  

  
 