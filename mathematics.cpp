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

vector<int> primeList,Phi( R+1, 0 );	//Phi[i] С��i������i���ʵ��������ĸ���
vector<bool> isPrime( R+1, true );
void getPrimes( int R){  // O(n)���ӶȻ�ȡС�ڵ���R���������ɼ���ŷ������Phi��
	for( int i(2) ; i <= R ; i++ ){  // ö��M �������S = M*p pΪS����С��������M��Ȼ��p����
		if( isPrime[i] ){
			primeList.push_back( i );
			//Phi[i] = i-1;	// phi(i) = i-1 if i is prime
		}
		for( int j(0) ; j < primeList.size() ; j++ ){			
			int k = i*primeList[j] ;
			if( k > R )	break;
			isPrime[k] = false;			
			//if( !(i % primeList[j] ) ){		// Phi(i*j) = j * Phi(i) if j is prime factor of i 
			//	Phi[k] = Phi[i] * primeList[j];
			//	break;
			//}
			//else
			//	Phi[k] = Phi[i]*Phi[primeList[j]];	// Phi(i*j) = Phi(i) * Phi(j) if i and j are coprime
		}
	}
}

vector<int> primeDecomposition( LL n ){ // ������n������Ψһ�Էֽ�
	// primeList ��������
	int i(0);
	vector<int> ret;
	while( primeList[i] <= n ){
		ret.push_back(0);
		while( !n%primeList[i] ){
			n/=primeList[i];
			ret[i]++;
		}
		i++;
	}
	return ret;
}
LL getDividors(LL num){  // ����� num ��Լ���ĸ���O(sqrt(num)) �ĸ��Ӷ�
    LL count = 0;  
    for(LL i=1;i*i<=num;i++){  
        if(num%i==0){  
            if (i!=num/i) {  
                count += 2;  
            }else {  
                count++;  
            }  
        }  
    }  
    return count;  
}  

LL gcd( LL a, LL b ){
	return a%b==0 ? b: gcd(b,a%b);
}

void num2bin( int num , int bit, char* s){ // 10������numת��Ϊbitλ�����Ʊ�ʾ���ַ���
	s[bit] = '\0';
	int i = bit-1;
	while( i >= 0 ){
		s[i] = num&1 ? '1':'0';
		num = num>>1;
		i--;
	}
}

int main(){
    freopen("in.txt","r",stdin);
    freopen("out.txt","w",stdout);
	
    return 0;
}