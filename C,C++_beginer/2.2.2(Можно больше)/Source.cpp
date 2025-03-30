#include<iostream>
#include<cstdarg>
#include <algorithm>
using namespace std;
class Animals
{
private:
	bool type;
	float weight;
public:
	Animals(bool nt = false, float nw = 0) : type(nt), weight(nw)
	{
		if (nt) cout << "Hi, Buvol! Tvoi ves " << nw << "kg" << endl;
		else cout << "Hi, Wolk! Tvoi ves " << nw << "kg" << endl;
	}
	Animals(Animals& src) : type(src.type), weight(src.weight)
	{
		cout << "Hi, copia!" << endl;
	}
	~Animals()
	{
		if (type) cout << "RIP Buivol vesom " << weight << "kg" << endl;
		else cout << "RIP Wolk vesom " << weight << "kg" << endl;
	}

	Animals& operator=(const Animals& src2)
	{
		weight = src2.weight;
		type = src2.type;
		return *this;
	}
	Animals razborki(int k, Animals& one, ...)
	{
		Animals total;
		va_list pick, pick1;
		va_start(pick, & one);
		va_start(pick1,& one);
		//	const va_list pick2 = pick;
		if (k > 2) cout << "Neverno" << endl;
		if (k == 1)
		{
			total.type = one.type && type;
			if (total.type == true)//Условие для буйволов
			{
				total.weight = max(weight, one.weight);
				total.type = 1;
				cout << "Rodilsya Byuvol!!! Ves ego " << total.weight << "kg" << endl;
				return total;
			}
			else //поиск буйвола и волка
				if (type == 1)
				{
					cout << "seli Buyvola! " << "Ves bil: " << weight << endl;
					one.weight += weight;
					cout << "Wolk teper vesit " << one.weight << "kg" << endl;
					this->~Animals();
				}
				else
				{
					if (one.type == 1)
					{
						cout << "seli Buyvola! " << "Ves bil: " << one.weight << endl;
						this->weight += one.weight;
						cout << "Wolk teper vesit " << weight << "kg" << endl;
						one.~Animals();
					}
					else
					{
						total.weight = min(weight, one.weight);
						total.type = 0;
						cout << "Rodilsya Wolk! Ves ego " << total.weight << "kg" << endl;
						return total;
					}

				}

		}
		else
		{
			//va_list pick;
				//va_start(pick, k);
			if (va_arg(pick1, Animals).type == 1 && one.type == 1 && type == 0)
			{
				cout << "Wolka vesom " << weight << "kg zatoptali byivoli!";
				weight = 0;
				//this->~Animals();
			}
			pick1 = pick;
			if (va_arg(pick1, Animals).type == 1 && one.type == 0 && type == 1)
			{
				cout << "Wolka vesom " << one.weight << "kg zatoptali byivoli!";
				//one.~Animals();
				one.weight = 0;
			}
			pick1 = pick;
			if (va_arg(pick1, Animals).type == 0 && one.type == 1 && type == 1)
			{
				cout << "Wolka vesom " << weight << "kg zatoptali byivoli!";
				//va_arg(pick, Animals).~Animals();
				va_arg(pick1, Animals).weight = 0;
			}
			pick1 = pick;
			if (((va_arg(pick1, Animals).type || one.type || type) == 0) || (va_arg(pick, Animals).type && one.type && type) == 1)
			{
				cout << "Vtroem razmnozhatsya nelzya!";
			}
			//	cout << va_arg(pick, Animals).type;
		}
	}
	float weig() const { return weight; }
	bool ty() const { return type; }
	void Zapolni()
	{
		cout << "Vid zhivotnogo (Wolk/Buffalo): ";
		string s;
		cin >> s;
		if (s == "Wolk") type = 0;
		if (s == "Buffalo") type = 1;
		cout << endl << "Ves: ";
		cin >> weight;
	}

};

ostream& operator<<(ostream& os, const Animals& src)
{
	if (src.ty() == 1) os << endl << "Buffalo " << endl;
	else os << endl << "Wolk " << endl;
	os << "Ves: " << src.weig() << endl << flush;
	return os;
}
class Massive
{
	Animals* mass;
	int len;
public:
	Massive(int N = 0,Animals mass_n=0) : mass(0), len(0)
	{
		if (N > 0) mass = new Animals[N];
		if (mass) len = N;

	}
	Massive(const Massive& temp) : mass(0), len(0)
	{
		if (temp.len > 0) mass = new Animals[temp.len];
		if (mass)
		{
			len = temp.len;
			for (int i = 0; i < len; i++)
			{
				mass[i] = temp.mass[i];
			}
		}
	}

	int size() const { return len; }

	Animals& operator [] (int index)
	{
		return mass[index];
	}

	~Massive()
	{
		cout << "RIP: ";
		for (int i = 0; i < len; i++)
		{
			cout << mass[i];
		}

		delete[] mass;
	}
	void insert(Animals& An)
	{
		Animals* tmp;
		tmp = new Animals[len];
		for (int i = 0; i < len; i++)
		{
			tmp[i] = mass[i];
		}
		mass = new Animals[len + 1];
		for (int i = 0; i < len; i++)
		{
			mass[i] = tmp[i];
		}
		mass[len] = An;
		len += 1;
	}
	void erase(int index, int k)
	{
		Animals* tmp;
		if (index >= len) cout << "Oshibochka" << endl;
		else
		{
			if (index + k > len - 1)
			{
				tmp = new Animals[index + 1];
				cout << "Massive ukorochen";
				len -= index + 1;
				for (int i = 0; i < index; i++)
				{
					tmp[i] = mass[i];
				}
				delete[] mass;
				mass = new Animals[index + 1];
				for (int i = 0; i < index; i++)
				{
					mass[i] = tmp[i];
				}
				delete[] tmp;

			}
			else
			{
				tmp = new Animals[len - k];
				for (int i =0;i<index;i++)
				{
					tmp[i] = mass[i];
				}
				for (int i = index+k,j=index; i < len; i++,j++)
				{
					tmp[j] = mass[i];
				}
				delete[] mass;
				mass = new Animals[len - k];
				for (int i = 0; i < len - k; i++)
				{
					mass[i] = tmp[i];
				}
				delete[] tmp;
				len -= k;
			}


		}

	}
	void Videli(int n)
	{
		mass = new Animals[n];
		len = n;
	}
};

int main()
{
	Animals C(0,20);
	Massive A, B;
	Animals* P;
	P = new Animals[2];
	for (int i = 0; i < 2; i++)
	{
		Animals R(0, 100);
		P[i] = R;
	}
	Massive F(sizeof(P) / sizeof(Animals), *P);
	int n,index,k;
	cout << "Vvedite kilochestvo zhivotnih: ";
	cin >> n;
	B.Videli(n);
	cout << "Vvedifte zhovotnih: ";
	for (int i = 0; i < n; i++)
	{
		B[i].Zapolni();
	}
	cout << "Zhivotnoe, kotoroe nyzno vstavit: ";
	C.Zapolni();
	B.insert(C);
	cout << "MASSIVE: ";
	for (int i = 0; i < B.size(); i++)
	{
		cout << B[i];
	}
	cout << "Vvedite index s kotorogo nyzno ydalit: ";
	cin >> index;
	cout << "Vvedite kolichestvo ydalennih yacheek: ";
	cin >> k;
	B.erase(index, k);
	cout << "MASSIVE: ";
	for (int i = 0; i < B.size(); i++)
	{
		cout << B[i];
	}

return 0;
}