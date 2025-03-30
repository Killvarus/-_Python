#include <iostream>
using namespace std;
#include<cstdarg>
#include <algorithm>

class Animals
{
private: 
bool type;
float weight;
public:
	Animals(bool nt = false, float nw = 0) : type(nt), weight(nw)
	{
		//if (nt) cout << "Hi, Buvol! Tvoi ves " << nw << "kg"<<endl;
		//else cout << "Hi, Wolk! Tvoi ves " << nw << "kg" << endl;
	}
	Animals(Animals& src) : type(src.type), weight(src.weight)
	{
		//cout << "Hi, copia!" << endl;
	}
	~Animals()
	{
		//if (type) cout << "RIP Buivol vesom " << weight << "kg" << endl;
		//else cout << "RIP Wolk vesom " << weight << "kg" << endl;
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
		va_list pick,pick1;
		va_start(pick, &one);
		va_start(pick1, &one);
	//	const va_list pick2 = pick;
		if (k > 2) cout << "Neverno" << endl;
		if (k == 1)
		{
			total.type = one.type&&type;
			if (total.type == true)//Условие для буйволов
			{
				total.weight = max(weight, one.weight);
				total.type = 1;
				cout << "Rodilsya Byuvol!!! Ves ego "<<total.weight<<"kg"<<endl;
				return total;
			}
			else //поиск буйвола и волка
				if (type == 1)
				{	
					cout << "seli Buyvola! " <<"Ves bil: " <<weight<< endl;
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
						cout << "Rodilsya Wolk! Ves ego " << total.weight<<"kg"<< endl;
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
		cout <<endl<< "Ves: ";
		cin >> weight;
	}

};

ostream& operator<<(ostream& os, const Animals& src)
	{
		if (src.ty() == 1) os << endl << "Buffalo " << endl;
		else os <<endl<< "Wolk " << endl;
		os <<"Ves: "<<src.weig() << endl << flush;
		return os;
	}
istream& operator>> (istream& is, Animals r)
{
	float wei;
	bool type; 
	is >> type;
	char c; 
	is >> c;
	if (c == ';') is >> wei;
	r = Animals(type, wei);
	return is;
}

int main() 
{
	int N;
	Animals* ONE;
	cout << "Vvedite kolichestvo zhivotnih" << endl;
	cin >> N;
	ONE = new Animals[N];
	cout << "Vvedite zhivotnih"<<endl;
	for (int i = 0; i < N; i++)
	{
		//cin >> ONE[i];
		ONE[i].Zapolni();
		cout<<endl;
		
	}
	switch (N)
	{
	case 1: cout << "Odno zhivotnoe skuchaet!!" << ONE[0]; break;
		case 2: ONE[0].razborki(1, ONE[1]);
		case 3: ONE[0].razborki(2, ONE[1], ONE[2]);

	}
	return 0;
}