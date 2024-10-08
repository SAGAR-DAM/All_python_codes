#include <iostream>
#include <chrono>
#include <ctime>

using namespace std;
using namespace std::chrono;

int main()
{
	// Get current time as a time_point
	system_clock::time_point tpoint = system_clock::now();
	
	// Convert to a time_t
	time_t tt = system_clock::to_time_t(tpoint);
	
	// Convert to local time
	tm* t = localtime(&tt);
	// Convert to readable format
	char buffer[80] = { 0 };
	strftime(buffer, sizeof(buffer), "%H:%M:%S", t);
	// Write the time to the console
	cout << buffer << endl;

	// You might get a security-related error or warning on the call to localtime() above.
	// With Microsoft Visual C++ you can use the safe version called localtime_s().
	// On Linux you can use localtime_r().
	// For example, for Visual C++ you can implement the above as follows:
	//// Convert to local time
	//tm t;
	//localtime_s(&t, &tt);
	//// Convert to readable format
	//char buffer[80] = { 0 };
	//strftime(buffer, sizeof(buffer), "%H:%M:%S", &t);
	//// Write the time to the console
	//cout << buffer << endl;

	return 0;
}
