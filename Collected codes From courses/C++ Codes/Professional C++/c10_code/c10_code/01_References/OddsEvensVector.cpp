#include <iostream>
#include <vector>
using namespace std;

void printVec(const vector<int>& arr)
{
	for (auto& i : arr) {
		cout << i << " ";
	}
	cout << endl;
} 

void separateOddsAndEvens(const vector<int>& arr,
						  vector<int>& odds, vector<int>& evens)
{
	int numOdds = 0, numEvens = 0;
	for (auto& i : arr) {
		if (i % 2 == 1) {
			++numOdds;
		} else {
			++numEvens;
		}
	}

	odds.reserve(numOdds);
	evens.reserve(numEvens);

	for (auto& i : arr) {
		if (i % 2 == 1) {
			odds.push_back(i);
		} else {
			evens.push_back(i);
		}
	}
}

int main()
{
	vector<int> vecUnSplit = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
	vector<int> odds, evens;

	separateOddsAndEvens(vecUnSplit, odds, evens);
	printVec(odds);
	printVec(evens);

	return 0;
}
