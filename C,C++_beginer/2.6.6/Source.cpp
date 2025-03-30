#include <iostream>
using namespace std;
class Queue {
    double* data;
    int capacity;
    int front;
    int rear;

public:
    Queue() {
        data = new double[1];
        capacity = 1;
        front = 0;
        rear = 0;
    }

    void push(double element) {
        if (rear == capacity) {
            double* temp = new double[2 * capacity];
            for (int i = 0; i < capacity; ++i) {
                temp[i] = data[i];
            }
            delete[] data;
            data = temp;
            capacity *= 2;
        }
        data[rear++] = element;
    }

    double top() {
        if (front == rear) {
            cout << "Queue is empty!" <<endl;
            return -1;
        }
        return data[front];
    }

    void pop() {
        if (front == rear) {
            cout << "Queue is empty!" <<endl;
            return;
        }
        for (int i = 0; i < rear - 1; ++i) {
            data[i] = data[i + 1];
        }
        rear--;
    }

    void destruct() {
        delete[] data;
        front = 0;
        rear = 0;
        capacity = 1;
    }
    void printQueue() {
        if (front == rear) {
            std::cout << "Queue is empty" << std::endl;
            return;
        }
        std::cout << "Queue: ";
        for (int i = front; i < rear; ++i) {
            std::cout << data[i] << " ";
        }
        std::cout << std::endl;
    }
};

int main() {
    Queue queue;
    queue.push(10.5);
    queue.push(20.7);
    queue.push(30.3);
    queue.printQueue();
    cout << "Top element: " << queue.top() << endl;
    queue.pop();
    queue.printQueue();
    cout << "Top element after pop: " << queue.top() << endl;
    queue.destruct();
    return 0;
}