#include <algorithm>
#include <vector>
#include <iostream>
using namespace std;

// Function template to populate a container of ints.
// The container must support push_back().
template<typename Container>
void populateContainer(Container& cont)
{
	int num;

	while (true) {
		cout << "Enter a number (0 to quit): ";
		cin >> num;
		if (num == 0) {
			break;
		}
		cont.push_back(num);
	}
}

int main()
{
	vector<int> vec1, vec2;

	populateContainer(vec1);

	vec2.resize(vec1.size());
	auto endIterator = copy_if(cbegin(vec1), cend(vec1),
		begin(vec2), [](int i){ return i % 2 == 0; });
	vec2.erase(endIterator, end(vec2));
	for (const auto& i : vec2) { cout << i << " "; }

	cout << endl;

	return 0;
}
