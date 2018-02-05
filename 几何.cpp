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
	// 计算圆C(圆心为c) 与三角形 cAB 相交的面积，请确保线段AB与圆不相交（穿过圆周），利用参数inner传递
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
	if( dcmp( fabs(C1.r-C2.r)-d) > 0 ) return 0; // 内含

	double a = angle( C2.c-C1.c );	// 向量C1C2的极角
	double da = acos( (C1.r*C1.r+d*d-C2.r*C2.r)/(2*C1.r*d) );// 余弦定理求 C1C2 到C1P1的角

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

// 判断线段交
bool SegmentProperIntersection(Point a1, Point a2, Point b1, Point b2) {
	double c1 = Cross(a2 - a1, b1 - a1), c2 = Cross(a2 - a1, b2 - a1),
		c3 = Cross(b2 - b1, a1 - b1), c4 = Cross(b2 - b1, a2 - b1);
	return dcmp(c1)*dcmp(c2) < 0 && dcmp(c3)*dcmp(c4) < 0; // 只是端点触碰不算
}
bool OnSegment(Point p, Point a1, Point a2) {	// proper on 
	return dcmp(Cross(a1 - p, a2 - p)) == 0 && dcmp(Dot(a1 - p, a2 - p)) < 0; // 考虑端点,则<=
}

// 判断点是否在三角形内
bool pointInTriangle(Point& p, Point& v1, Point& v2, Point& v3) {
	// 判断点与三角形顶点连线与三角形对应的两条边位置关系
	if (Cross(v2 - v1, p - v1)*Cross(v3 - v1, p - v1) <= 0
		&& Cross(v3 - v2, p - v2)*Cross(v1 - v2, p - v2) <= 0)
		return true;
	return false;
}

// 判断两个三角形是否有公共面积
bool triangleIntersection(Point* tri1, Point* tri2) {
	// 判断两个三角形是否相互包含
	if (pointInTriangle(tri1[0], tri2[0], tri2[1], tri2[2])
		&& pointInTriangle(tri1[1], tri2[0], tri2[1], tri2[2])
		&& pointInTriangle(tri1[2], tri2[0], tri2[1], tri2[2]))	return true;
	if (pointInTriangle(tri2[0], tri1[0], tri1[1], tri1[2])
		&& pointInTriangle(tri2[1], tri1[0], tri1[1], tri1[2])
		&& pointInTriangle(tri2[2], tri1[0], tri1[1], tri1[2]))	return true;
	// 如果不是相互包含，必然有两个线段有交（非端点交）
	for (int i(0); i < 3; i++)
		for (int j(0); j < 3; j++) {
			if (SegmentProperIntersection(tri1[i], tri1[(i + 1) % 3], tri2[j], tri2[(j + 1) % 3]))
				return true;
		}
	return false;
}

// 凸包
bool cmp(Point a, Point b) { if (a.x == b.x) return a.y < b.y;     return a.x < b.x; }
int ConvexHull(Point* p, int n, Point* ch) {
	sort(p, p + n, cmp);
	int m = 0;
	Point vec1, vec2;
	for (int i = 0; i < n; i++) {
		while (m > 1) {
			vec1 = ch[m - 1] - ch[m - 2];
			vec2 = p[i] - ch[m - 1];
			if (Cross(vec1, vec2) <= 0)
				m--;
			else break;
		}
		ch[m++] = p[i];
	}
	int k = m;
	for (int i = n - 2; i >= 0; i--) {
		while (m > k) {//一开始已经有1个点   
			vec1 = ch[m - 1] - ch[m - 2];
			vec2 = p[i] - ch[m - 2];
			if (Cross(vec1, vec2) <= 0)
				m--;
			else break;
		}
		ch[m++] = p[i];
	}
	if (n > 1) m--;
	return m;
}