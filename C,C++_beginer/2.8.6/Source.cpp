#include <iostream>
#include <string>
using namespace std;
class Calculator {
public:
    double add(double a, double b) {
        return a + b;
    }

    double subtract(double a, double b) {
        return a - b;
    }

    double multiply(double a, double b) {
        return a * b;
    }

    double divide(double a, double b) {
        if (b == 0) {
            throw runtime_error("Zero error!");
        }
        return a / b;
    }
};

int main() {
    setlocale(LC_ALL, "Russian");
    Calculator calc;
    string input;

    while (true) {
        cout << "������� �������� (+, -, *, /) � ��� ����� (����� ������), ��� ������� 'quit' ��� ������: ";
        cin >> input;

        if (input == "quit") {
            cout << "����� ������." << endl;
            break;
        }

        char op;
        double a, b;
        try {
            cin >> a >> op >> b;

            double result;
            switch (op) {
            case '+':
                result = calc.add(a, b);
                break;
            case '-':
                result = calc.subtract(a, b);
                break;
            case '*':
                result = calc.multiply(a, b);
                break;
            case '/':
                result = calc.divide(a, b);
                break;
            default:
                throw invalid_argument("������������ ��������!");
            }

            cout << "���������: " << result << endl;
        }
        catch (const exception& e) {
            cout << "������: " << e.what() << endl;
        }

    }

    return 0;
}