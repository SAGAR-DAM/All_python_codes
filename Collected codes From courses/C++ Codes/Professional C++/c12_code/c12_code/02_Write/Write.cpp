#include <iostream>
#include <cstring>

using namespace std;

void rawWrite(const char* data, int dataSize)
{
	cout.write(data, dataSize);
}

int main()
{
	const char* test = "hello there\n";
	rawWrite(test, strlen(test));

	return 0;
}
