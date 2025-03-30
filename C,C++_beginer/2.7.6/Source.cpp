#include <iostream>
#include <fstream>
#include <vector>
#include <string>
using namespace std;
void transposeMatrix(const string& inputFile, const string& outputFile) {
    ifstream input(inputFile);
    ofstream output(outputFile);

    if (!input.is_open()) {
        cout << "Не удалось открыть входной файл." <<endl;
        return;
    }

    vector<vector<int>> matrix;
    string line;

    while (getline(input, line)) {
        vector<int> row;
        size_t pos = 0;
        while ((pos = line.find(' ')) != string::npos) {
            row.push_back(stoi(line.substr(0, pos)));
            line.erase(0, pos + 1);
        }
        row.push_back(stoi(line));
        matrix.push_back(row);
    }

    vector<vector<int>> transposedMatrix(matrix[0].size(), vector<int>(matrix.size()));
    for (size_t i = 0; i < matrix.size(); ++i) {
        for (size_t j = 0; j < matrix[i].size(); ++j) {
            transposedMatrix[j][i] = matrix[i][j];
        }
    }

    for (int i = 0; i < transposedMatrix.size();i++){
        for (int j = 0; j < transposedMatrix[i].size();j++) {
            output << transposedMatrix[i][j] << " ";
        }
        output <<endl;
    }

    input.close();
    output.close();
}

int main() {
    setlocale(LC_ALL, "Russian");
    transposeMatrix("input.txt", "output.txt");
    cout << "Матрица успешно транспонирована и записана в файл output.txt." << endl;
    return 0;
}