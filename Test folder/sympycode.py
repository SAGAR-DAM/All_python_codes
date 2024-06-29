from sympy import *
x, y, z, e, a, b = symbols('x y z e a b')
#str(Integral(sqrt(1/x), x))
'Integral(sqrt(1/x), x)'
#print(Integral(sqrt(1/x), x))
#Integral(sqrt(1/x), x)

#srepr(Integral(sqrt(1/x), x))
#"Integral(Pow(Pow(Symbol('x'), Integer(-1)), Rational(1, 2)), Tuple(Symbol('x')))"

pprint(Integral(sqrt(1/x), x), use_unicode=True)

pprint(Integral(sqrt(1/pow(x,2)), x), use_unicode=False)

pprint(pow(e,-pow((x-a),2)/pow(b,2)),use_unicode=False)

pprint(pow(a,2)/(pow(b,2)+pow(x,2)),use_unicode=False)



pprint(Integral(sqrt(1/(pow(x,(Integral(sqrt(1/x), x)))+(pow(a,2)/(pow(b,2)+pow(x,(pow(a,2)/(pow(b,2)+pow(x,(pow(a,2)/(pow(b,2)+pow(x,2))))))))))), x), use_unicode=False)