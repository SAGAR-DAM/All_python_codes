#include <iostream>

using namespace std;

void performTask()
{
    static bool initialized = false;
    
	if (!initialized) {
        cout << "initializing\n";
        // Perform initialization.
        initialized = true;
    }

	// Perform the desired task.
}

int main()
{
	performTask();
	performTask();

	return 0;
}
