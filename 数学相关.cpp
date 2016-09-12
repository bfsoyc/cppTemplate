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

const int e = 1000000007;
typedef long long LL;

LL fermat( LL a, LL d ){
	LL ans = 1;
	while(d){
		if( d&1 ) ans = ans*a%e;
		a = a * a % e;
		d >>= 1;
	}
	return ans;
}

LL invPivot( LL a){		//a����Ϊ����
	int m = e;
	//���� a ģm����Ԫ��m��������ֱ���÷���С������⣺ a^(m-2) % m
	//return fastExponentation(a, m-2);
	return fermat( a, m-2 );
}

vector<LL> matrixCross( vector<LL> a, vector<LL> b){ // 2X2����˷�
	LL M[] = {
		a[0]*b[0]+a[1]*b[2], a[0]*b[1]+a[1]*b[3],
		a[2]*b[0]+a[3]*b[2], a[2]*b[1]+a[3]*b[3]
	};
	for( int i(0) ; i < 4; i++ ) M[i]%=e;
	return vector<LL>(M, M+4);
}

vector<LL> matrixPower( vector<LL>& p, LL d ){ //�ݹ�д��
	if( d == 1 ){
		return p;
	}
	if( d%2 ){
		return matrixCross( matrixPower(p,d-1), p );
	}
	else{
		vector<LL> tmp = matrixPower(p, d/2 );
		return matrixCross( tmp, tmp );
	}
}

int main(){
    freopen("in.txt","r",stdin);
    freopen("out.txt","w",stdout);
	
    return 0;
}