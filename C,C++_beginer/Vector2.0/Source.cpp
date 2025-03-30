#include<iostream>
using namespace std;
class Vector
{
	double* mass;
	int size, capacity;

public:
	Vector(int size_n = 0, double* mass_n = 0) : mass(0), capacity(size_n), size(size_n)
	{
		cout << "Poyavilsya Vector " << endl;
		mass = new double[size_n];
		if (size_n && !mass_n)
		{

			for (int i = 0; i < size_n; i++)
			{
				mass[i] = 0;
			}
		}
		else
		{
			if (size_n && mass_n)
			{
				for (int i = 0; i < size_n; i++)
				{
					mass[i] = mass_n[i];
				}
			}
			if (!size_n && mass_n) throw size_n;
		}
	}
	Vector(Vector& src) : size(src.size), capacity(src.capacity)
	{
		mass = new double[src.capacity];
		cout << "Hi, copia!" << endl;
		for (int i = 0; i < src.size; i++)
		{
			mass[i] = src.mass[i];
		}
	}
	~Vector()
	{
		cout << "Vector RIP" << endl;
		for (int i = 0; i < size; i++)
		{
			cout << mass[i] << "\t";
		}
		cout << endl;

	}
	Vector& operator=(const Vector& src2)
	{
		size = src2.size;
		capacity = src2.capacity;
		mass = new double[size];
		for (int i = 0; i < size; i++)
		{
			mass[i] = src2.mass[i];
		}
		return *this;
	}
	int size_t()
	{
		return { size };
	}
	int capacity_t()
	{
		return capacity;
	}
	double& operator [] (int index)
	{
		return mass[index];
	}
	void resize(int new_capacity)
	{
		double* mass_n;
		if (capacity > new_capacity)
		{
			mass_n = new double[new_capacity];
			for (int i = 0; i < min(size, new_capacity); i++)
			{
				mass_n[i] = mass[i];
			}
			capacity = new_capacity;
			size = min(new_capacity, size);
			mass = new double[new_capacity];
			for (int i = 0; i < size; i++)
			{
				mass[i] = mass_n[i];
			}
			delete[] mass_n;
		}
		else
		{
			mass_n = new double[new_capacity];
			for (int i = 0; i < size; i++)
			{
				mass_n[i] = mass[i];
			}
			capacity = new_capacity;
			delete[] mass;
			mass = new double[new_capacity];
			for (int i = 0; i < size; i++)
			{
				mass[i] = mass_n[i];
			}
			delete[] mass_n;

		}

	}
	Vector& operator+=(Vector& other)
	{
		if (capacity >= other.capacity)
		{
			for (int i = 0; i < min(other.size, size); i++)
			{
				mass[i] += other.mass[i];
			}
		}
		else
		{
			Vector C;
			C = other;
			for (int i = 0; i < capacity; i++)
			{
				C.mass[i] += mass[i];
			}
			*this = C;
		}
		return *this;
	}
	Vector& operator-=(const Vector& other)
	{
		if (capacity >= other.capacity)
		{
			for (int i = 0; i < min(other.size, size); i++)
			{
				mass[i] -= other.mass[i];
			}
		}
		else
		{
			Vector C;
			C = other;
			for (int i = 0; i < capacity; i++)
			{
				C.mass[i] -= mass[i];
			}
			*this = C;
		}
		return *this;
	}
	Vector operator+(const Vector& other)
	{
		Vector result;
		result.capacity = max(other.capacity, capacity);
		result.mass = new double[result.capacity];
		for (int i = 0; i < max(size, other.size); ++i)
		{
			result.mass[i] = mass[i] + other.mass[i];
		}
		return result;
	}
	Vector operator-(const Vector& other)
	{
		Vector result;
		result.capacity = max(other.capacity, capacity);
		result.mass = new double[result.capacity];
		for (int i = 0; i < max(size, other.size); ++i)
		{
			result.mass[i] = mass[i] - other.mass[i];
		}
		return result;
	}
	Vector operator*(double chislo)
	{
		Vector result;
		for (int i = 0; i < size; ++i) {
			result.mass[i] = mass[i] * chislo;
		}
		return result;
	}
	Vector operator/(double chislo)
	{
		Vector result;
		for (int i = 0; i < size; ++i) {
			result.mass[i] = mass[i] / chislo;
		}
		return result;
	}
	void push_back_dbl(double element)
	{
		if (size == capacity)
		{
			Vector cpy{ *this };
			capacity *= 2;
			size++;
			delete[] mass;
			mass = new double[capacity];
			for (int i = 0; i < size-1; i++)
			{
				mass[i] = cpy.mass[i];
			}
			mass[size - 1] = element;

		}
		else
		{
			mass[size] = element;
			size++;
		}
	}
	void pop_back()
	{
		if (size)
		{
			capacity = size = size - 1;
			Vector cpy{ *this };
			delete[] mass;
			mass = new double[size];
			*this = cpy;
		}
		
	}
	void insert_dbl(int ind,double chislo)
	{
		if (ind > size) throw ind;
		if (size == capacity)
		{
			Vector cpy{ *this };
			delete [] mass;
			size++;
			capacity *= 2;
			mass = new double[size];
			for (int i = 0; i < ind;i++)
			{
				mass[i] = cpy.mass[i];
			}
			mass[ind] = chislo;
			for (int i = ind + 1; i <= cpy.size; i++)
			{
				mass[i] = cpy.mass[i-1];
			}
		}
		else 
		{
			Vector cpy{ *this };
			size++;
			for (int i = 0; i < ind; i++)
			{
				mass[i] = cpy.mass[i];
			}
			mass[ind] = chislo;
			for (int i = ind + 1; i <= cpy.size; i++)
			{
				mass[i] = cpy.mass[i - 1];
			}
		}
	}
	
};
ostream& operator<<(ostream& os, Vector& src)
	{
	os << "Massive(Vector chisel): " << endl;
	for (int i = 0; i < src.size_t(); i++)
	{
		os << src[i] << "\t";
	}
	}
	int main() {
		double* mass = new double[5];
		for (int i = 0; i < 5; i++)
		{
			mass[i] = 1;
		}
		try
		{
			Vector A(5, mass), B;
			B= A;
			A.resize(10);
			for (int i = 0; i < 10; i++)
			{
				A.push_back_dbl(2);
			}
			A += B;
			for (int i = 0; i < A.size_t(); i++)
			{
				cout << A[i];
			}
			A.pop_back();
			for (int i = 0; i < A.size_t(); i++)
			{
				cout << A[i];
			}
			A.resize(900);
			A.insert_dbl(2, 999);
			for (int i = 0; i < A.size_t(); i++)
			{
				cout << A[i];
			}
			cout << A;
		}
		catch (const int ex)
		{
			cout << "LALALALA" << ex;
		}
		
		
		return 0;
	}
