#include <vector>
#include <iostream>
using namespace std;
template <class T> void sort(T arr[], int size)
{
    for (int i = 0; i < size; i++)
    {
        for (int k = i + 1; k < size; k++)
        {
            if (arr[k] < arr[i])
            {
                T tmp(arr[i]);
                arr[i] = arr[k];
                arr[k] = tmp;
            }
        }

    }
}
template <class T> void sort_vstavka(T mass[], int size)
{
    T El;
    int place;
    for (int i = 1; i < size; i++)
    {
        El = mass[i];
        place = i - 1;
        for (place = i - 1; place >= 0 && mass[place] > El; place--)
        {
            mass[place + 1] = mass[place];

        }
        mass[place + 1] = El;
    }
}

class String :public vector<char> {
public:
    String() : vector<char>{ 0 } {}

    friend ostream& operator<<(ostream& os, String& src);
    friend istream& operator>>(istream& is, String& src);

};
ostream& operator<<(ostream& os, String& src)
{
    for (int i = 0; i < src.size(); i++)
    {
        cout << src[i];
    }
    cout << '\t';
    return os;
}
istream& operator>>(istream& is, String& src)
{
    string tmp;
    cin >> tmp;
    copy(tmp.begin(), tmp.end(), back_inserter(src));
    return is;
}
int main()
{
    setlocale(LC_ALL, "Russian");
    String* str;
    int strs;
    cout << "Введите количество строк: ";
    cin >> strs;

    str = new String[strs];
    for (int i = 0; i < strs; i++)
    {
        cout << "Введите строку " << i << ": ";
        cin >> str[i];
        cout << endl;
    }

    sort_vstavka(str, strs);
    cout << "Отсортированные строки: " << endl;
    for (int i = 0; i < strs; i++)
    {
        cout << str[i];
    }

    return 0;
}