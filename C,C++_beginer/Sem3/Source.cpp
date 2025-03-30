#include<iostream>
using namespace std;
class Chiselki 
{
	double* mass;
	int N;
public:
	Chiselki(int N_n = 0, double* mass_n = 0) :N(N_n),mass(0)
	{
		{
			cout << "Poyavilis chiselki "<<endl;
				if (N_n && !mass_n)
				{
					mass = new double[N_n];
					N = N_n;
					for (int i = 0; i < N_n; i++)
					{
						mass[i] = 0;
					}
				}
				else
				{
					if (N_n && mass_n)
					{
						N = N_n;
						for (int i = 0; i < N; i++)
						{
							mass[i] = mass_n[i];
						}
					}
					if (!N_n && mass_n) throw N_n;
				}
			}
		}
	Chiselki(Chiselki& src) : N(src.N)
	{
		cout << "Hi, copia!" << endl;
		for (int i = 0; i < N; i++)
		{
			src.mass[i] = mass[i];
		}
	}
	~Chiselki()
	{
		cout << "Chiselki skonchalis"<<endl;
		for (int i = 0; i < N; i++)
		{
			cout << mass[i] << "\t";
		}

	}
	Chiselki& operator=(const Chiselki &src2)
	{
		N = src2.N;
		mass = new double[N];
		for (int i = 0; i < N; i++)
		{
			mass[i] = src2.mass[i];
		}
		return *this;
	}
	float size()
	{
		return N;
	}
	void Videli(int n)
	{
		mass = new double[n];
		N = n;
	}
	double& operator [] (int index)
	{
		return mass[index];
	}
	Chiselki& operator+=(const Chiselki& other)
	{
		for (int i = 0; i < N; i++) 
		{
			mass[i] += other.mass[i];
		}
		return *this;
	}
	Chiselki& operator-=(const Chiselki& other)
	{
		for (int i = 0; i < N; i++) {
			mass[i] -= other.mass[i];
		}
		return *this;
	}
	//Chiselki& operator=(const Chiselki& src)
	//{
		//for (int i = 0; i < N; i++)
	//	{
		//	mass[i] = src.mass[i];
		//}
	//	/return *this;
	//}


	Chiselki operator+(const Chiselki& other)
	{
		Chiselki result;
		result.N = max(other.N, N);
		result.mass = new double[result.N];
		for (int i = 0; i < result.N; ++i) 
		{
			result.mass[i] = mass[i] + other.mass[i];
		}
		return result;
	}

	Chiselki operator-(const Chiselki& other)
	{
		Chiselki result;
		result.N = max(other.N, N);
		result.mass = new double[result.N];
		for (int i = 0; i < result.N; ++i)
		{
			result.mass[i] = mass[i] - other.mass[i];
		}
		return result;
	}
	Chiselki operator*(double chislo)
	{
		Chiselki result;
		for (int i = 0; i < N; ++i) {
			result.mass[i] = mass[i] * chislo;
		}
		return result;
	}

	Chiselki operator/(double chislo)
	{
		Chiselki result;
		for (int i = 0; i < N; ++i) {
			result.mass[i] = mass[i] / chislo;
		}
		return result;
	}
};
int main()
{
	double* mass;
	mass = new double[5];
	for (int i = 0; i < 5; i++)
	{
		mass[i] = 1;
	}
	try
	{
		Chiselki A(0, mass);

		cout << 1;



	}
	catch (const int ex)
	{
		cout << "LALALALA" << ex;
	}



	return 0;

}