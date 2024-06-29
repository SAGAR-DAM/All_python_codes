class myclass:     #creates a parent class
    x=5             #different class values can be assigned
    y=[1,2,3]
    def f(x,y):         #a class function
        return(x+y[2])
    z=f(x,y)            #inter class function application


class myclass1:
    def g(x,y):
        return(x**y) 
