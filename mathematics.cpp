#define _CRT_SECURE_NO_WARNINGS
//#pragma comment(linker, "/STACK:1024000000,1024000000")
#include<iostream>
#include<cstdio>
#include<vector>
#include<map>
#include<string>
#include<queue>
#include<cstring> // strlen
#include<cmath>
#include<algorithm>
#include<stack>
#include<limits.h>
#include<set>
#include<string>
using namespace std;

const int e = 1000000007;
typedef long long LL;
const double PI = acos( -1.0 );

LL fermat( LL a, LL d ){
	LL ans = 1;
	while(d){
		if( d&1 ) ans = ans*a%e;
		a = a * a % e;
		d >>= 1;
	}
	return ans;
}

LL invPivot( LL a){		//a可以为负数
	int m = e;
	//返回 a 模m的逆元，m是素数，直接用费马小定理求解： a^(m-2) % m
	//return fastExponentation(a, m-2);
	return fermat( a, m-2 );
}

vector<LL> matrixCross( vector<LL> a, vector<LL> b){ // 2X2矩阵乘法
	LL M[] = {
		a[0]*b[0]+a[1]*b[2], a[0]*b[1]+a[1]*b[3],
		a[2]*b[0]+a[3]*b[2], a[2]*b[1]+a[3]*b[3]
	};
	for( int i(0) ; i < 4; i++ ) M[i]%=e;
	return vector<LL>(M, M+4);
}

vector<LL> matrixPower( vector<LL>& p, LL d ){ //递归写法
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

vector<int> primeList,Phi( R+1, 0 );	//Phi[i] 小于i并且与i互质的正整数的个数
vector<bool> isPrime( R+1, true );
void getPrimes( int R){  // O(n)复杂度获取小于等于R的素数（可计算欧拉函数Phi）
	for( int i(2) ; i <= R ; i++ ){  // 枚举M 任意合数S = M*p p为S的最小质因数，M必然与p互质
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

vector<int> primeDecomposition( LL n ){ // 对整数n作素数唯一性分解
	// primeList 是素数表
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
LL getDividors(LL num){  // 获得数 num 的约数的个数O(sqrt(num)) 的复杂度
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

void num2bin( int num , int bit, char* s){ // 10进制数num转化为bit位二进制表示的字符串
	s[bit] = '\0';
	int i = bit-1;
	while( i >= 0 ){
		s[i] = num&1 ? '1':'0';
		num = num>>1;
		i--;
	}
}

//复数结构体
struct complex
{
    double r,i;
	complex(double _r = 0.0,double _i = 0.0){  r = _r; i = _i; }
    complex operator +(const complex &b){
        return complex(r+b.r,i+b.i);
    }
    complex operator -(const complex &b){
        return complex(r-b.r,i-b.i);
    }
    complex operator *(const complex &b){
        return complex(r*b.r-i*b.i,r*b.i+i*b.r);
    }
};
// DFT ( naive recursive )
// X(k) = F1(k) + W(k,N)F2(k);  both F1 and F2 are of period of N/2 
// the inverse matrix of W is 1/N of the conjugate matrix of W ?
vector<complex> DFT( vector<complex> x , int type = 1){
	int N = x.size();
	int hN = N/2;
	if( N == 1 ){
		return x;
	}
	complex WN(cos(-2*type*PI/N), sin(-2*type*PI/N) );
	vector<complex> oddX,evenX;
	for( int i(0); i < N; i+=2 ){
		oddX.push_back( x[i] );
		evenX.push_back( x[i+1] );
	}
	vector<complex> F1 = DFT( oddX, type ), F2 = DFT( evenX, type ), X(N);
	complex w(1,0); // 为什么在没有重载=操作符的情况下， w = 1 能达到同样效果
	for( int k = 0 ; k < N ; k++ ){
		X[k] = F1[ k%hN ] + w*F2[ k%hN ];
		if( type==-1 ){ 
			// 应该是错的  IDFT(k) 不应该等于 IDFT( oddX ) + W(k,N)*IDFT(evenX)
			X[k].r /= N, X[k].i /= N; // ?
		}
		w = w*WN;
	}
	return X;
}
// 我们希望把奇数下标的数放到前半段，偶数下标的放到后半段，再对这两个半段递归处理
// 然而这样处理无疑会多操作很多次，下面参考网上的模板，未参透
void change(complex *y,int len)
{
    int i,j,k;
    for(i = 1, j = len/2;i < len-1; i++){
        if(i < j)swap(y[i],y[j]); //交换互为小标反转的元素，i<j保证交换一次 
		//i做正常的+1，j左反转类型的+1,始终保持i和j是反转的
        k = len/2;
        while( j >= k){
            j -= k;
            k /= 2;
        }
        if(j < k) j += k;
    }
}
// 值得注意的是，如果卷积结果出现很大的数，可能有较大的精度误差，但是相对值的大小的不变的
// 傅里叶变换对 x[n](*)y[n] <-> X(k)*Y(k)   (*)表示圆周卷积：注意对y序列有一个反转操作。
void DFT( complex *x, int L, int type = 1 ){
	change( x, L );
	for( int N = 2; N <= L ; N = N << 1 ){ // 从较短的长度递推较长的长度
		complex wn( cos(-type*2*PI/N), sin(-type*2*PI/N) );
		for( int i = 0 ; i < L; i += N ){ // 计算长度为N的各个block
			complex w(1,0);
			for( int j = i; j < i+N/2; j++ ){
				complex u = x[j], t = x[j+N/2];
				x[j] = u+w*t;
				x[j+N/2] = u-w*t;
				w = w*wn;
			}
		}
	}
	if( type==-1 )
		for( int i(0);  i < L; i++ ) 
			// 为什么最后才除? 看上递归部分代码 
			// 而且仅仅考虑实部（原序列一定为实数？）。
			x[i].r /= L;
}


int main(){
    freopen("in.txt","r",stdin);
    freopen("out.txt","w",stdout);
	
    return 0;
}