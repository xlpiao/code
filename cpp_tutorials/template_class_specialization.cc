#include <cstring>
#include <iostream>
using namespace std;

template <typename T>
class Point {
 private:
  T xpos, ypos;

 public:
  Point(T x = 0, T y = 0) : xpos(x), ypos(y) {}
  void ShowPosition() const {
    cout << '[' << xpos << ", " << ypos << ']' << endl;
  }
};

template <typename T>
class SimpleDataWrapper  // 클래스 템플릿 SimpleDataWrapper가 정의
{
 private:
  T mdata;

 public:
  SimpleDataWrapper(T data)
      : mdata(data)  // 하나의 데이터를 저장, 이 데이터에 담긴 정보 출력
  {}
  void ShowDataInfo(void) { cout << "Data: " << mdata << endl; }
};

template <>
class SimpleDataWrapper<char*>  // 클래스 템플릿 SimpleDataWrapper를 char* 형에
                                // 대해서 특수화(문자열을 저장하기 위한 것)
{
 private:
  char* mdata;

 public:
  SimpleDataWrapper(char* data) {
    mdata = new char[strlen(data) +
                     1];  // 동적할당 기반의 생성자와 소멸자를 별도로 정의
    strcpy(mdata, data);
  }
  void ShowDataInfo(void) {
    cout << "String: " << mdata << endl;
    cout << "Length: " << strlen(mdata) << endl;  // 문자열의 길이정보 출력
  }
  ~SimpleDataWrapper() { delete[] mdata; }
};

template <>
class SimpleDataWrapper<Point<int> >  // Point<int>형에 대해서 특수화 : 자료형의
                                      // 이름을 Point<int>가 대신
{
 private:
  Point<int> mdata;

 public:
  SimpleDataWrapper(int x, int y) : mdata(x, y) {}
  void ShowDataInfo(void) { mdata.ShowPosition(); }
};

int main(void) {
  SimpleDataWrapper<int> iwrap(170);
  // int형에 대해서는 특수화가 진행되지 않음
  // 컴파일 될 때 템플릿 클래스인 SimpleDataWrapper<int>가 만들어지고, 이
  // 클래스를 기반으로 객체 생성
  iwrap.ShowDataInfo();

  SimpleDataWrapper<char*> swrap("Class Template Specialization");
  // char* 형에 대해서 특수화 진행 -> 별도의 클래스 생성되지 않고, 위에 정의된
  // 템플릿 클래스의 객체 생성
  swrap.ShowDataInfo();

  SimpleDataWrapper<Point<int> > poswrap(3, 7);
  // Point<int> 형에 대해서 특수화 진행
  poswrap.ShowDataInfo();
  return 0;
}
