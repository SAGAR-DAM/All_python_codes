#include <iostream>
#include <string>

using namespace std;

void handleValue(int value)
{
	cout << "Integer: " << value << endl;
}
void handleValue(double value)
{
	cout << "Double: " << value << endl;
}
void handleValue(const string& value)
{
	cout << "String: " << value << endl;
}



// First version using pass-by-value
template<typename T>
void processValues(T arg)	// Base case
{
	handleValue(arg);
}

template<typename T1, typename... Tn>
void processValues(T1 arg1, Tn... args)
{
	handleValue(arg1);
	processValues(args...);
}



// Second version using pass-by-rvalue-reference
template<typename T>
void processValuesRValueRefs(T&& arg)
{
	handleValue(std::forward<T>(arg));
}

template<typename T1, typename... Tn>
void processValuesRValueRefs(T1&& arg1, Tn&&... args)
{
	handleValue(std::forward<T1>(arg1));
	processValuesRValueRefs(std::forward<Tn>(args)...);
}



int main()
{
	processValues(1, 2, 3.56, "test", 1.1f);
	cout << endl;
	processValuesRValueRefs(1, 2, 3.56, "test", 1.1f);

	return 0;
}
