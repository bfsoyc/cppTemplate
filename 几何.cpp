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


const double eps = 1e-8;
 
int dcmp(double x) {
    if (fabs(x) < eps) return 0;
    else return x < 0 ? -1 : 1;
}

// T是自定义精度类型
typedef struct Point{  //  不必要时不要用long long，乘法运算慢很多 
    T x,y; 
    Point( T x=0 , T y=0):x(x),y(y){}
}Vector; 

Point operator - (const Point& a,const Point& b){ return Point(a.x-b.x,a.y-b.y);}  //常量引用能快一点点 
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
	double delta = f*f - 4*e*g; // 判别式
	
	double t1,t2;
	if( dcmp(delta) < 0 ) return 0; //相离
	if( dcmp(delta) == 0 ){ // 相切
		t1 = t2 = -f/(2*e);
	}
	// 相交
	t1 = (-f-sqrt(delta))/(2*e);
	t2 = (-f+sqrt(delta))/(2*e);
	// 当点A在圆内， 点B在圆外时， t1,t2必定一个>0 一个小于0
	// 并且大于0的一点对应圆与 线段AB的交点。
	return Point(A.x+max(t1,t2)*v.x, A.y+max(t1,t2)*v.y);
}

double includedAngle( Vector& a, Vector& b){
	// |a|,|b| 不等于0
	double c = Dot(a,b)/(Length(a)*Length(b));
	return acos(max(-1.0,min(c,1.0))); // refine the value of c 
}

double TriagnleCircleIntersectionArea( Circle& C, Point& A, Point& B, bool inner){
	// 计算圆C(圆心为c) 与三角形 cAB 相交的面积，请确保线段AB完全不在圆内或完全不在圆外，利用参数inner传递
	if( inner )
		return 0.5*Cross(A-C.c, B-C.c ); // 这里计算有向面积，方向自行排序
	else{
		double theta = includedAngle( A-C.c, B-C.c );
		return dcmp(Cross(A-C.c, B-C.c ))*0.5*theta*C.r*C.r;
	}
}

// 计算向量极角
double angle( Vector v ){ return atan2(v.y,v.x); }
// 两圆相交
int getCircleCircleIntersection( Circle C1, Circle C2, vector<Point>& sol, vector<double>& agl ){
	double d = Length( C1.c-C2.c );
	if( dcmp(d) == 0 ){
		if( dcmp(C1.r - C2.r )==0 ) return -1; // 两圆重合
		return 0; // 圆心重合半径不同
	}
	if( dcmp( C1.r+C2.r-d ) < 0 ) return 0; // 外离
	if( dcmp( fabs(C1.r-C2.r)-d) > 0 ) return 0; // 相切

	double a = angle( C2.c-C1.c );	// 向量C1C2的极角
	double da = acos( (C1.r*C1.r+d*d-C2.r*C2.r)/(2*C1.r*d) );
	// C1C2 到C1P1的角
	Point p1 = C1.point(a-da), p2 = C1.point(a+da);

	agl.clear();
	//sol.push_back(p1);
	agl.push_back(a-da);
	if( dcmp(p1.x-p2.x)==0 && dcmp(p1.y-p2.y)==0 ) return 1;
	//sol.push_back( p2);
	agl.push_back(a+da);
	// C1上从p1到p2的一段弧在C2内
	return 2;
}