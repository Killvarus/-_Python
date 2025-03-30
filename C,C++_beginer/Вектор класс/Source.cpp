#include <iostream>

#define KOL 5
using namespace std;
class Vector {
private:
    double x[KOL];

public:
    Vector() 
    {
        cout << "Hi!";
        for (int i = 0; i < KOL; ++i) {
            x[i] = 0;
        }
    }
    Vector(const Vector& src) 
    {
        cout << "Kuku";
        for (int i = 0; i < KOL; i++) {
            x[i] = src.x[i];
        }
    }

    ~Vector() 
    {
        cout << "RIP: (";
        for (int i = 0; i < KOL; i++)
        {
            if (i != KOL - 1)
            {
                cout << x[i] << ";";
            }
            else cout << x[i] << ")";
        }
    }
    Vector& operator+=(const Vector& other) 
    {
        for (int i = 0; i < KOL; i++) {
            x[i] += other.x[i];
        }
        return *this;
    }
    Vector& operator-=(const Vector& other) 
    {
        for (int i = 0; i < KOL; i++) {
            x[i] -= other.x[i];
        }
        return *this;
    }
    Vector& operator=(const Vector& src)
    {
        for (int i = 0; i < KOL; i++)
        {
            x[i] = src.x[i];
        }
        return *this;
    }

  
    Vector operator+(const Vector& other) const 
    {
        Vector result;
        for (int i = 0; i < KOL; ++i) {
            result.x[i] = x[i] + other.x[i];
        }
        return result;
    }

    Vector operator-(const Vector& other) const 
    {
        Vector result;
        for (int i = 0; i < KOL; ++i) {
            result.x[i] = x[i] - other.x[i];
        }
        return result;
    }
    Vector operator*(double chislo) const 
    {
        Vector result;
        for (int i = 0; i < KOL; ++i) {
            result.x[i] = x[i] * chislo;
        }
        return result;
    }

    Vector operator/(double chislo) const
    {
        Vector result;
        for (int i = 0; i < KOL; ++i) {
            result.x[i] = x[i] / chislo;
        }
        return result;
    }
    
    
    void Zapolni() 
    {
        cout << "Vvedite " << KOL << " koordinat vectora:" <<endl;
        for (int i = 0; i < KOL; i++) {
            cin >> x[i];
        }
    }
    void Vivedi()
    {
        cout << "Vector: (";
        for (int i = 0; i < KOL; i++)
        {
            if (i != KOL - 1)
            {
                cout << x[i] << ";";
            }
            else 
            {
                cout << x[i] << ")";
            }
        }
    }

};

int main() {
    Vector A,B,C;
    char c;
    cout << "Vvedite perviy vector: ";
    A.Zapolni();
    cout << "Vvedite vtoroy vector: ";
    B.Zapolni();
    cout << "Vvedite operaciu";
    cin >> c;
    switch (c)
    {
    case '+': C = A + B; break;
    case '-': C = A - B; break;


    }
    C.Vivedi();

    return 0;
}