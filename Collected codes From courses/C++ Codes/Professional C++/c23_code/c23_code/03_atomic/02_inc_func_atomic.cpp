#include <iostream>
#include <vector>
#include <thread>
#include <functional>
#include <chrono>
#include <atomic>

void func(std::atomic<int>& counter)
{
	for (int i = 0; i < 100; ++i) {
		++counter;
		std::this_thread::sleep_for(std::chrono::milliseconds(1));
	}
}

int main()
{
	std::atomic<int> counter(0);
	std::vector<std::thread> threads;

	for (int i = 0; i < 10; ++i) {
		threads.push_back(std::thread{ func, std::ref(counter) });
	}

	for (auto& t : threads) {
		t.join();
	}

	std::cout << "Result = " << counter << std::endl;

	return 0;
}
