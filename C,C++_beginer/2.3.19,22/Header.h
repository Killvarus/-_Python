#include<iostream>
using namespace std;
class Vector
{
protected:
	double* mass;
	int size, capacity;

public:
	Vector(int size_n = 0, double* mass_n = 0) : capacity(size_n), size(size_n)
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
	Vector(Vector& src) : size(src.size), capacity(src.capacity),mass(0)
	{
		cout << "Hi, copia!" << endl;
		if (size)
		{
			mass = new double[capacity];
			
			for (int i = 0; i < size; i++)
			{
				mass[i] = src.mass[i];
			}
		}
		
	}
	~Vector()
	{
		cout << "Vector RIP" << endl;
		cout << *this;
		cout << endl;
		delete [] mass;

	}
	Vector& operator=(const Vector& src2)
	{
		size = src2.size;
		capacity = src2.capacity;
		delete [] mass;
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
	void size_n(int k) { size = k; }
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
		if (capacity && size)
		{
			if (capacity > new_capacity)
			{
				mass_n = new double[new_capacity];
				size = min(size, new_capacity);
				for (int i = 0; i < size; i++)
				{
					mass_n[i] = mass[i];
				}
				capacity = new_capacity;
				delete mass;
				mass = new double[capacity];
				for (int i = 0; i < size; ++i)
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
		else
		{
			delete[] mass;
			mass = new double[new_capacity];
			capacity = new_capacity;

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
	Vector& operator*=(const Vector& other)
	{
		if (capacity >= other.capacity)
		{
			for (int i = 0; i < min(other.size, size); i++)
			{
				mass[i] *= other.mass[i];
			}
		}
		else
		{
			Vector C;
			C = other;
			for (int i = 0; i < capacity; i++)
			{
				C.mass[i] *= mass[i];
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
		Vector result{*this};
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
			delete mass;
			mass = new double[capacity];
			for (int i = 0; i < size - 1; i++)
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
	void insert_dbl(int ind, double chislo)
	{
		if (ind > size) throw ind;
		if (size == capacity)
		{
			Vector cpy( *this );
			size++;
			capacity *= 2;
			this->resize(capacity);
			for (int i = 0; i < ind; i++)
			{
				mass[i] = cpy.mass[i];
			}
			mass[ind] = chislo;
			for (int i = ind + 1,j=ind; j < cpy.size&&i<capacity; i++,j++)
			{
				mass[i] = cpy.mass[j];
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
	friend ostream& operator<<(ostream& os, Vector& src);
	double* size_zv(int ind)
	{
		double* t = (mass+ind);
		return t;
	}
	friend istream& operator>>(istream& is, Vector& src);

};
ostream& operator<<(ostream& os, Vector& src)
{
	os << "Massive(Vector chisel): " << endl;
	for (int i = 0; i < src.size_t(); i++)
	{
		os << src[i] << "\t";
	}
	return os;
}
istream& operator>>(istream& is, Vector& src)
{
	int k;
	cout << "Vvedite razmer Vectora: " << endl;
	is >> k;
	src.resize(k);
	src.size_n(k);
	cout << "VVedite chisla: " << endl;
	for (int i = 0; i < src.capacity_t(); i++)
	{
		is >> src[i];
	}
	return is;
}
class Matrix :public Vector
{
	int line, col;
public:

	Matrix(int line_n = 0, int col_n = 0) : line(line_n), col(col_n), Vector{ 0 }
	{
		if (line && col)
		{
			this->resize(line * col);
			//matrix = new Vector[line];
			for (int i = 0; i < line*col; i++)
			{
				push_back_dbl(0);
			}
		}
		

	}
	~Matrix() 
	{
		//Vector::~Vector();
		cout << "Matrix RIP"<<endl;
	}
	Matrix(Matrix& src): line(src.line),col(src.col)
	{
		if (line && col)
		{
			this->operator=(src);
		}
	}
	Matrix &operator=(Matrix& src)
	{
		line = src.line;
		col = src.col;
		resize(col * line);
		Vector::operator=(src);
		return *this;
		}
	Matrix& operator=(const Matrix& src)
	{
		line = src.line;
		col = src.col;
		delete[] mass;
		mass = new double[col * line];
		Vector::operator=(src);
		return *this;
	}
	double* operator[] (int line)
	{
		return size_zv(line);
	}
	friend ostream& operator<<(ostream& os, Matrix& src);
	friend istream& operator>>(istream& is, Matrix& src);
	int line_t() { return line; }
	int col_t() { return col; }
	void vvod(int line, int col)
	{
		resize(line * col);
		size_n(line * col);
		this->col = col;
		this->line = line;

	}
	Matrix operator+(Matrix& other)
	{
		Matrix result{*this};
		if (col == other.col && line == other.line)
		{
			for (int i = 0; i < size; i += col)
			{
				for (int j = 0; j < col; j++)
				{
					result[i][j] += other[i][j];
				}
			}
		}
		else throw (other.line, other.col);
		return result;
	}
	Matrix operator-(Matrix& other)
	{
		Matrix result{ *this };
		if (col == other.col && line == other.line)
		{
			for (int i = 0; i < size; i += col)
			{
				for (int j = 0; j < col; j++)
				{
					result[i][j] -= other[i][j];
				}
			}
		}
		else throw (other.line, other.col);
		return result;
	}
	Matrix &operator*(double chislo)
	{
		Matrix result( *this );
		result.Vector::operator=(result.Vector::operator*(chislo));
		return result;
	}
	Matrix operator*(Matrix& src)
	{
		Matrix result(line,src.col_t());
		for (int i = 0,l=0; i <result.size_t(); i+=src.col_t(),l+=col_t())
		{
			for (int j = 0; j < src.col_t(); j++) 
			{
				for (int k = 0; k < col_t(); k++)
				{
					result[i][j] += (*this)[l][k]* src[k * src.col_t()][j];
					cout << result[i][j];
				}
			}
		}
		return result;
	}
	Matrix operator*(Vector& src)
	{
		Matrix tmp(1, src.size_t());
		tmp.Vector::operator=(src);
		Matrix result;
		result =( (*this) * (tmp));
		return result;

	}
	Matrix operator*=(Vector& src)
	{
		*this = ((*this) * src);
		return *this;
	}
	Matrix operator*=(double chislo)
	{
		*this = *this * chislo;
		return *this;
	}
	Matrix operator+=(Matrix& src)
	{
		*this = src + *this;
		return *this;
	}
	Matrix operator -=(Matrix& src)
	{
		*this = src - *this;
		return *this;
	}
};
ostream& operator<<(ostream& os, Matrix& src)
{
	os << "Matrix: " << endl;
	for (int i = 0; i < src.size_t(); i += src.col_t())
	{
		for (int j = 0; j < src.col_t(); j++)
		{
			os << src[i][j] << "\t";
		}
		os << "\n"<<endl;
	}
	return os;
}
istream& operator>>(istream& is, Matrix& src)
{	
	if (src.col_t() && src.line_t())
	{
		cout << "Vvedite chisla v matricy: " << endl;
		for (int i = 0; i < src.size_t(); i += src.col_t())
		{
			cout << "Stroka ¹" << i / src.col_t() << ": " << endl;
			for (int j = 0; j < src.col_t(); j++)
			{
				is >> src[i][j];
			}
		}
	}
	else 
	{
		int col, line;
		cout << "Vvedite kolichestvo strok: " << flush;
		is >> line;
		cout << "Vvediet  stolbcov: ";
		is >> col;
		src.vvod(line, col);
		cout << "Vvedite chisla v matricy: " << endl;
		for (int i = 0; i < src.size_t(); i += src.col_t())
		{
			cout << "Stroka " << i / src.col_t() << ": " << endl;
			for (int j = 0; j < src.col_t(); j++)
			{
				is >> src[i][j];
			}
		}
	}
	
	return is;
}