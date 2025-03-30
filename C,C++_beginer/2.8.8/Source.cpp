#include <iostream>
#include <cmath>
#include <stdexcept>

using namespace std;

// Функция уравнения f(x)
double equation(double x) {
    return x * x - 4; // Пример: x^2 - 4 = 0
}

// Функция поиска корня методом деления отрезка пополам
double findRoot(double a, double b, double epsilon = 1e-6) {
    if (equation(a) * equation(b) > 0) {
        throw runtime_error("Root not found on interval");
    }

    while (b - a > epsilon) {
        double mid = (a + b) / 2;
        if (equation(mid) == 0) {
            return mid;
        }
        else if (equation(mid) * equation(a) < 0) {
            b = mid;
        }
        else {
            a = mid;
        }
    }

    return (a + b) / 2;
}

int main() {
    double a, b;

    cout << "Enter a(start of int): ";
    cin >> a;
    cout << "Enter b(end of int): ";
    cin >> b;

    try {
        double root = findRoot(a, b);
        cout << "Root f(x) = 0 on int [" << a << ", " << b << "] equals: " << root << endl;
    }
    catch (const exception& e) {
        cerr << "Error!: " << e.what() << endl;
    }

    return 0;
}