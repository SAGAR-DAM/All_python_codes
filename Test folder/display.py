# importing limited functions
# defined in calc.py

import testa as testa
import arbitrary as arb
 
def main():
    d=arb.add(1, 2)
    print(d)
    
    x=[1,2,3,4,5,6]
    y=testa.square(x)
    
    print(y)
    
if __name__=='__main__':
    main()
