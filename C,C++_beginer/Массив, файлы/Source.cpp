#include"Header1.h"
extern int i, j, t, v, y;
int main() {
	srand(time(NULL));
	int mass[kol];
	for (int i = 0; i < kol; i++) {
		mass[i] =1+rand()%1500;
	}
	cout << "Pervonachalniy massiv: " << endl;
	for (int i = 0; i < kol; i++) { cout << mass[i] << "     "; }
	sort(mass, kol);
	cout <<endl<< "Otsortirovanniy massiv: " << endl;
	for (int i = 0; i < kol; i++) { cout << mass[i]<<"      "; }
	cout <<endl<<endl<<"Kolichestvo perestanovok ravno: "<< v << endl <<"Kolichestvo sravneniy ravno^ "<< y<<endl;
	return 0;
}