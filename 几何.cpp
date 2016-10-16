#define _CRT_SECURE_NO_WARNINGS
//#pragma comment(linker, "/STACK:1024000000,1024000000")
#include<iostream>
#include<cstdio>
#include<vector>
#include<map>
#include<string>
#include<queue>
#include<cstring>
#include<cmath>
#include<algorithm>
#include<stack>
#include<limits.h> // INT_MAX
#include<set>
#include<string>
using namespace std;
typedef long long LL;
const int maxn = 105;

const double eps = 1e-8;
 
int dcmp(double x) {
    if (fabs(x) < eps) return 0;
    else return x < 0 ? -1 : 1;
}

// T���Զ��徫������
typedef struct Point{  //  ����Ҫʱ��Ҫ��long long���˷��������ܶ� 
    T x,y; 
    Point( T x=0 , T y=0):x(x),y(y){}
}Vector; 

Point operator - (const Point& a,const Point& b){ return Point(a.x-b.x,a.y-b.y);}  //���������ܿ�һ��� 
T Cross( const Point& a,const Point& b ){ return a.x*b.y - a.y*b.x ;}; 
T Dot( const Point& a,const Point& b ){ return a.x*b.x + a.y*b.y; }; 
double Length( const Point& a ){ return sqrt( ((double)a.x)*a.x + ((double)a.y)*a.y ); } 

struct Circle{
	Point c;
	double r;
	Circle( Point c, double r):c(c),r(r){}
	Point point(double a){
		return Point(c.x+cos(a)*r, c.y+sin(a)*r);
	}
};
double PolygonArea( Point* p, int n ){
	double area = 0;
	for( int i = 1; i < n-1 ; i++ )
		area += Cross(p[i]-p[0], p[i+1]-p[0]);
	return area/2;
}



Point getLineCircleIntersection( Point& A, Point& B, Circle C){
	Vector v = B-A;
	double a = v.x, b = A.x-C.c.x, c = v.y, d = A.y - C.c.y;
	double e = a*a + c*c, f = 2*(a*b+c*d), g = b*b+d*d-C.r*C.r;
	double delta = f*f - 4*e*g; // �б�ʽ
	
	double t1,t2;
	if( dcmp(delta) < 0 ) return 0; //����
	if( dcmp(delta) == 0 ){ // ����
		t1 = t2 = -f/(2*e);
	}
	// �ཻ
	t1 = (-f-sqrt(delta))/(2*e);
	t2 = (-f+sqrt(delta))/(2*e);
	// ����A��Բ�ڣ� ��B��Բ��ʱ�� t1,t2�ض�һ��>0 һ��С��0
	// ���Ҵ���0��һ���ӦԲ�� �߶�AB�Ľ��㡣
	return Point(A.x+max(t1,t2)*v.x, A.y+max(t1,t2)*v.y);
}

double includedAngle( Vector& a, Vector& b){
	// |a|,|b| ������0
	double c = Dot(a,b)/(Length(a)*Length(b));
	return acos(max(-1.0,min(c,1.0))); // refine the value of c 
}

double TriagnleCircleIntersectionArea( Circle& C, Point& A, Point& B, bool inner){
	// ����ԲC(Բ��Ϊc) �������� cAB �ཻ���������ȷ���߶�AB��ȫ����Բ�ڻ���ȫ����Բ�⣬���ò���inner����
	if( inner )
		return 0.5*Cross(A-C.c, B-C.c ); // ����������������������������
	else{
		double theta = includedAngle( A-C.c, B-C.c );
		return dcmp(Cross(A-C.c, B-C.c ))*0.5*theta*C.r*C.r;
	}
}