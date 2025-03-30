#include <iostream>
#include <time.h>
#include <stdlib.h>
using namespace std;

int main() {
	srand(time(NULL));
	int mass[8];
	for (int i = 0; i < 8; i++) {
		mass[i] = rand();
		cout << mass[i] << endl;
}

	return 0;
}