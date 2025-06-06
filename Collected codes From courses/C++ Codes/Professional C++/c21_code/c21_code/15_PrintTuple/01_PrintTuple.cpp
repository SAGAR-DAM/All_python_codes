#include <iostream>
#include <string>
#include <tuple> 

using namespace std;

template<int n, typename TupleType>
class tuple_print
{
public:
	tuple_print(const TupleType& t) {
		tuple_print<n - 1, TupleType> tp(t);
		cout << get<n - 1>(t) << endl;
	}
};

template<typename TupleType>
class tuple_print<0, TupleType>
{
public:
	tuple_print(const TupleType&) { }
};

int main()
{
	using MyTuple = tuple<int, string, bool>;
	MyTuple t1(16, "Test", true);
	tuple_print<tuple_size<MyTuple>::value, MyTuple> tp(t1);

	return 0;
}
