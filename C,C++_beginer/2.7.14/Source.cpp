#include <iostream>
#include <fstream>
#include <string>
#include <io.h>
#include <map>
using namespace std;
void file(string in, string out) {
	ifstream fin(in);
	map<char, int> m;
	if (fin.is_open()) {
		cout << "Файл для чтения открыт" << endl;
		string str;
		while (!fin.eof()) {
			getline(fin, str);

			for (int i = 0; i < str.length(); i++) {
				m[str[i]]++;
				cout << str[i];
			}

		}
	}
	fin.close();
	map<char, int>::iterator it = m.begin();
	ofstream fout(out);
	if (fout.is_open()) {
		cout << "Файл для записи открыт" << endl;
		for (it; it != m.end(); it++) {
			if ((*it).first == 0 || (*it).first == ' ') {
				continue;
			}
			fout << (*it).first << " - " << (*it).second << endl;
		}
	}
	fout.close();
}
int main() {
	setlocale(LC_ALL, "rus");
	file("1.txt", "2.txt");
	system("pause");
	return (0);
}