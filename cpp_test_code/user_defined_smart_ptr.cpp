#include <iostream>
using namespace std;

class Point {
  private:
    int xpos, ypos;

  public:
    Point(int x = 0, int y = 0) : xpos(x), ypos(y) {
        cout << "Point Constructor" << endl;
    }
    ~Point() {
        cout << "Point Destructor" << endl;
    }
    void SetPos(int x, int y) {
        xpos = x;
        ypos = y;
    }
    friend ostream& operator<<(ostream& output_stream, const Point& pos);
};
ostream& operator<<(ostream& output_stream, const Point& pos) {
    cout << "Call operator<<:  ";
    output_stream << '[' << pos.xpos << ", " << pos.ypos << ']' << endl;
    return output_stream;
}

class SmartPtr {
  private:
    Point* pos_ptr_;

  public:
    SmartPtr(Point* pos_ptr) : pos_ptr_(pos_ptr) {
        cout << "SmartPtr Constructor" << endl;
    }

    Point& operator*() const    //스마트 포인터는 포인터처럼 동작하는 객체
    {
        return *pos_ptr_;
    }
    Point* operator->() const    // 두 함수의 정의는 필수
    {
        return pos_ptr_;
    }
    ~SmartPtr() {
        cout << "SmartPtr Destructor" << endl;
        delete pos_ptr_;
    }
};

int main(void) {
    SmartPtr sptr1(new Point(1, 2));    // Point 객체를 생성하면서, 동시에 스마트 포인터 SmartPtr 객체가 이를 가리키게끔함
    SmartPtr sptr2(new Point(2, 3));    // sptr1, sptr2, sptr3는 Point 객체를 가리키는 포인터처럼 동작
    SmartPtr sptr3(new Point(4, 5));
    cout << *sptr1;    // 포인터처럼 * 연산 진행
    cout << *sptr2;
    cout << *sptr3;

    sptr1->SetPos(10, 20);    // 포인터처럼 -> 연산 진행
    sptr2->SetPos(30, 40);
    sptr3->SetPos(50, 60);
    cout << *sptr1;
    cout << *sptr2;
    cout << *sptr3;

    return 0;
}
