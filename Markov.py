# CIS 410/510pm
# Homework #4
# Daniel Lowd
# May 2017
#
# TEMPLATE CODE
import sys
import tokenize
import itertools

# List of variable cardinalities is global, for convenience.
# NOTE: This is not a good software engineering practice in general.
# However, the autograder code currently uses it to set the variable 
# ranges directly without reading in a full model file, so please keep it
# here and use it when you need variable ranges!
var_ranges = []


#
# FACTOR CLASS -- EDIT HERE!
#

class Factor(dict):
    def __init__(self, scope_, vals_):
        self.scope = scope_
        self.vals = vals_
        # TODO -- ADD EXTRA INITIALIZATION CODE IF NEEDED

    def __mul__(self, other):
        """Returns a new factor representing the product."""
        # TODO -- PUT YOUR MULTIPLICATION CODE HERE!
        # BEGIN PLACEHOLDER CODE -- DELETE THIS! 
        new_scope = self.scope
        new_vals  = self.vals
        # END PLACEHOLDER CODE
        return Factor(new_scope, new_vals)

    def __rmul__(self, other):
        return self * other

    def __imul__(self, other):
        return self * other

    def __repr__(self):
        """Return a string representation of a factor."""
        rev_scope = self.scope[::-1]
        val = "x" + ", x".join(str(s) for s in rev_scope) + "\n"
        itervals = [range(var_ranges[i]) for i in rev_scope]
        for i,x in enumerate(itertools.product(*itervals)):
            val = val + str(x) + " " + str(self.vals[i]) + "\n"
        return val


#
# READ IN MODEL FILE
#

# Read in all tokens from stdin.  Save it to a (global) buf that we use
# later.  (Is there a better way to do this? Almost certainly.)
curr_token = 0
token_buf = []

def read_tokens():
    global token_buf
    for line in sys.stdin:
        token_buf.extend(line.strip().split())
    #print "Num tokens:",len(token_buf)

def next_token():
    global curr_token
    global token_buf
    curr_token += 1
    return token_buf[curr_token-1]

def next_int():
    return int(next_token())

def next_float():
    return float(next_token())

def read_model():
    # Read in all tokens and throw away the first (expected to be "MARKOV")
    read_tokens()
    s = next_token()

    # Get number of vars, followed by their ranges
    num_vars = next_int()
    global var_ranges;
    var_ranges = [next_int() for i in range(num_vars)]

    # Get number and scopes of factors 
    num_factors = int(next_token())
    factor_scopes = []
    for i in range(num_factors):
        scope = [next_int() for i in range(next_int())]
        # NOTE: 
        #   UAI file format lists variables in the opposite order from what
        #   the pseudocode in Koller and Friedman assumes. By reversing the
        #   list, we switch from the UAI convention to the Koller and
        #   Friedman pseudocode convention.
        scope.reverse()
        factor_scopes.append(scope)

    # Read in all factor values
    factor_vals = []
    for i in range(num_factors):
        factor_vals.append([next_float() for i in range(next_int())])

    # DEBUG
    print "Num vars: ",num_vars
    print "Ranges: ",var_ranges
    print "Scopes: ",factor_scopes
    print "Values: ",factor_vals
    return [Factor(s,v) for (s,v) in zip(factor_scopes,factor_vals)]


#
# MAIN PROGRAM
#

if __name__ == "__main__":
    factors = read_model()
    # Compute Z by brute force
    f = reduce(Factor.__mul__, factors)
    z = sum(f.vals)
    print "Z = ",z