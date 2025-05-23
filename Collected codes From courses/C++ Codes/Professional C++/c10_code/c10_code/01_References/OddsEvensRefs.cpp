#include <iostream>
using namespace std;

void printIntArr(const int arr[], int size)
{
	for (int i = 0; i < size; ++i) {
		cout << arr[i] << " ";
	}
	cout << endl;
} 

void separateOddsAndEvens(const int arr[], int size, int*& odds,
						  int& numOdds, int*& evens, int& numEvens)
{
	numOdds = numEvens = 0;
	for (int i = 0; i < size; ++i) {
		if (arr[i] % 2 == 1) {
			++numOdds;
		} else {
			++numEvens;
		}
	}

	odds = new int[numOdds];
	evens = new int[numEvens];
	
	int oddsPos = 0, evensPos = 0;
	for (int i = 0; i < size; ++i) {
		if (arr[i] % 2 == 1) {
			odds[oddsPos++] = arr[i];
		} else {
			evens[evensPos++] = arr[i];
		}
	}
}

int main()
{
	int unSplit[10] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
	int *oddNums, *evenNums;
	int numOdds, numEvens;

	separateOddsAndEvens(unSplit, 10, oddNums, numOdds, evenNums, numEvens);
	printIntArr(oddNums, numOdds);
	printIntArr(evenNums, numEvens);

	return 0;
}
