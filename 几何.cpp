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
#include<limits.h>
#include<set>
#include<string>
using namespace std;
typedef long long LL;
const int maxn = 105;

typedef struct Point{  //  不必要时不要用long long，乘法运算慢很多 
     LL x,y; 
     Point( int x=0 , int y=0):x(x),y(y){} 
}Vector; 

bool cmp( Point a , Point b ); 
Point operator - (const Point& a,const Point& b){ return Point(a.x-b.x,a.y-b.y);}  //常量引用能快一点点 
LL Cross( const Point& a,const Point& b ){ return a.x*b.y - a.y*b.x ;}; 
int Dot( const Point& a,const Point& b ){ return a.x*b.x + a.y*b.y; }; 
double Length( const Point& a ){ return sqrt( ((double)a.x)*a.x + ((double)a.y)*a.y ); } 

double PolygonArea( Point* p, int n ){
	double area = 0;
	for( int i = 1; i < n-1 ; i++ )
		area += Cross(p[i]-p[0], p[i+1]-p[0]);
	return area/2;
}