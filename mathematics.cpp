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
	LL ans = 1; // d = k_0*2^0+k_1*2^1+...+k_p*2^p; k_i为0或1下面d&1就是判断k_i的值
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

int primeList[R+1],primeNum; 
int Phi[R+1]; //Phi[i] 小于i并且与i互质的正整数的个数
bool isPrime[R+1]; // -1 表示true
void getPrimes( int R){  // O(n)复杂度获取小于等于R的素数（可计算欧拉函数Phi）
	primeNum = 0;
	memset(isPrime,-1,sizeof(isPrime));
	for( int i(2) ; i <= R ; i++ ){  // 枚举i=M 任意合数S = M*p p为S的最小质因数，M的最小质因素必然>=p,下面有分类谈论。
		if( isPrime[i] ){
			primeList[primeNum++] = i;
			//Phi[i] = i-1;	// phi(i) = i-1 if i is prime
		}
		for( int j(0) ; j < primeNum ; j++ ){			
			int k = i*primeList[j] ;
			if( k > R )	break;
			isPrime[k] = 0;			
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
		while( !(n%primeList[i]) ){
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

// 因子枚举
void dfs(int idx, int sum, vector<int> &pos, vector<int> &Pcnt, vector<int>& factors, LL& sq){
	if( sum > sq ) return;
	if( idx==pos.size() ){
		factors.push_back(sum);
		return;
	}
	int f(1);
	for( int i(0); i <= Pcnt[idx]; i++ ){
		dfs(idx+1, sum*f, pos, Pcnt, factors, sq);
		f *= primeList[pos[idx]];
	}
}
void getFactors( LL n, vector<int>& factors ){
	// 我们只需要枚举小于sqrt(n)的因子即可，若x是n的小于sqrt(n)的一个因子，自然地n/x是x的一个大于sqrt(n)的因子
	// 通过n的素因子的组合得到n的因子，对于大于sqrt(n)的素因子是不需考虑的。
	// primeList 是素数表
	LL sq = sqrt(n)+1; // LL for reason
	if( sq*sq > n ) sq--;
	vector<int> pos, Pcnt;
	int i(0);
	while( primeList[i] <= sq && primeList[i]<= n){
		if( !(n%primeList[i]) ){
			int cnt(0);
			while( !(n%primeList[i]) ){
				n/=primeList[i];
				cnt++;
			}
			Pcnt.push_back(cnt);
			pos.push_back(i);
		}
		i++;
	}
	// n的素因子个数（包括重复的）不会很多， 2^20已经达到10^6级别了。
	// 所以n的因子个数不会很多，假设n有7个因子2,7个因子3,7个因子5,也只有8^3种不同的组合
	factors.clear();
	dfs(0, 1, pos, Pcnt, factors, sq);
}

long long mod_mul(long long a, long long b, long long n) {
	// 模拟二进制乘法 计算 a*b % n 的值 
    long long res = 0;
    while (b) {
        if(b & 1)
            res = (res + a) % n;
        a = (a + a) % n;
        b >>= 1;
    }
    return res;
}

long long fastPowerMOD( long long a, long long u, long long MOD){
	// calculate a^u % MOD; 
	// tranform u to its binary form, consider its digit from lower bit to higher bit
	long long ret = 1;
	while( u ){
		if( u&1 )// ret = ret*a % MOD; // ret*a 可能溢出64位整数
			ret = mod_mul(ret, a, MOD );
		//a = a*a % MOD;
		a = mod_mul(a, a, MOD );
		u = u >> 1;
	}
	return ret;
}

// 检查一个数是否质数。 n > 2  复杂度O( aList.size()*logn )
vector<int> aList; // 如果n<2^64，只用选取a=2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37做测试
bool MillerRabin(long long n){
	if( n%2 == 0 ) return false; // n 为偶数
	// 如果 a^2 = p ( mod n )，那么 a = 1 (mod n ) 或 a = -1 (mod n )
	// 因为费马小定理： a^(n-1) = 1 (mod n ), 如果n-1是偶数，我们可以应用上式
	// a^((n-1)/2) = 1 (mod n ) 或 a^((n-1)/2) = -1 (mod n )
	// 如果 (n-1)/2 仍然为偶数 并且 a^((n-1)/2) = 1 (mod n ) 成立，那么还可以继续分解
	// 这样我们得到一系列应该成立的等式，若某一阶段不成立了，则n不是素数
	// 我们先找到的最小的a^u，再逐步扩大到a^(n-1)
	long long u = n-1; // u 为指数
	while( u%2==0 ) u>>=1;

	for( int i(0); i < aList.size(); i++ ){
		int a = aList[i]; if( a>=n ) continue;
		long long x = fastPowerMOD(a, u, n );
		while( u < n-1 ){ // 只需进行r-1次， 2^r*d = n-1 (d是奇数）
			//long long y = x*x % n;
			long long y = mod_mul( x, x, n );
			if( y==1 && x!=1 && x!=n-1 ){
				// y = x^2 = 1( mod n ), 但是 x != (1 or -1 )(mod n )违反二次检测定理 
				return false;
			}
			x = y, u <<= 1;
		}
		if( x != 1 ) // fermat 小定理
			return false;
	}
	return true;
}


//
LL gcd( LL a, LL b ){
	return a%b==0 ? b: gcd(b,a%b);
}

// 拓展欧几里得，得出 ax+by=gcd(a,b) 的一组解，（确保a，b>0，否则欧几里得算出的gcd可能为负数）
// 拓展到系解，很多时候需要最小整数解 x = x' + u*(b/d)， y = x' - u*(a/d), d= gcd(a,b)
// 若 c = t*gcd(a,b) 那么解系可以表示为 x = x'*t + u*(b/d)
// c++一般的编译器计算得到的余数r 是使得 m*k 与y的差不超过m的绝度值的最大的k下取得的r
void gcd(long long a, long long b, long long& d, long long& x, long long& y) {
	if (!b) { d = a, x = 1, y = 0; }
	else {
		gcd(b, a%b, d, y, x);
		y -= x*(a / b);
	}
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
// the inverse matrix of W is 1/N of the conjugate matrix of W ?( yes, the different colume in W is "perpendicular" to each other)
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
void change(complex *y,int len){
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
// 傅里叶变换对 x[n](*)y[n] <-> X(k)*Y(k)   (*)表示圆周卷积：注意对y序列有一个反转操作(0位置不变)。
void DFT( complex *x, int L, int type = 1 ){
	change( x, L );
	for( int N = 2; N <= L ; N = N << 1 ){ // 从较短的长度递推较长的长度
		complex wn( cos(-type*2*PI/N), sin(-type*2*PI/N) );
		for( int i = 0 ; i < L; i += N ){ // 计算长度为N的各个block
			complex w(1,0);
			for( int j = i; j < i+N/2; j++ ){
				complex u = x[j], t = x[j+N/2];
				x[j] = u+w*t;
				x[j+N/2] = u-w*t; // property
				w = w*wn;
			}
		}
	}
	if( type==-1 )
		for( int i(0);  i < L; i++ ) 
			// 为什么最后才除? 看上递归部分代码 
			// 而且仅仅考虑实部（原序列一定为实数？事实上作为题目的输入，一般还是整数）。
			x[i].r /= L;
}


// NTT: Number Theoretic Transforms 
// 定理如是说：	P是模数， omiga是模P意义下的（P-1）次单位原根， 那么对于n，如果n能整除(P-1)，即有: P = ksi*n + 1,
// 那么模P意义下的n次单位原根存在,等于 omiga^ksi =  omiga^((P-1)/n );
// NTT中我们要做多次 n 点数论变换， n 为 2的幂， 故一个能写成 alpha * 2^t + 1 形式的素数能够帮助我们快速找到计算过程所有需要的原根
// const LL P = 1945555039024054273LL; // 27 * (2 ^ 56), 1e18, g = 5  
const LL MOD = 50000000001507329LL; //190734863287 * 2 ^ 18 + 1, g = 3 
const int omiga = 3;
LL wn[20]; // wn[i]: 模P意义下的2^i次单位原根
void getWn() {
	for (int i = 1; i < 20; ++i) {
		int t = 1 << i;
		wn[i] = fastPowerMOD(omiga, (MOD - 1) / t, MOD);
	}
}
LL mul(LL x, LL y) {	// compute x*y % MOD in case x*y overflow
	return (x * y - (LL)(x / (long double)MOD * y + 1e-3) * MOD + MOD) % MOD;
}
void NTT(LL *x, int L, int type = 1) {
	change(x, L);	// change( LL*, int ）;
	getWn();
	int id(0);
	for (int N = 2; N <= L; N = N << 1) { // 从较短的长度递推较长的长度
		++id;
		for (int i = 0; i < L; i += N) { // 计算长度为N的各个block
			LL w = 1;
			for (int j = i; j < i + N / 2; j++) {
				LL u = x[j], t = x[j + N / 2];
				x[j] = u + mul(w, t);	// ensure u and the return value of mul() is less than MOD 
				if (x[j] > MOD) x[j] -= MOD;
				x[j + N / 2] = u - mul(w, t) + MOD;
				if (x[j + N / 2] > MOD) x[j + N / 2] -= MOD;
				w = mul(w, wn[id]);
			}
		}
	}
	if (type == -1) {
		for (int i(1); i < L / 2; i++) swap(x[i], x[L - i]);	// 与上一份DFT模板不同的是，只在这里区分正逆变换
		LL inv = invPivot(L);	// 计算长度L模MOD意义下的逆元
		for (int i(0); i < L; i++)
			x[i] = mul(x[i], inv);
	}
}

// 博弈论
// Sprague-Grundy，  sg[u] = min{ { i | i = 0,1,2... }/{sg[v]| v是u的后继局面} }
// SG 函数定义了Nim博弈的抽象模型
// 终结局面（先手必败局面) SG值为0， 非终结局面SG值>0
// 如果游戏能分成多个独立的子游戏,那么 SG[u] = SG[v_1] Xor SG[v_2] Xor...Xor[v_N], v_i 是多个独立的游戏所处的局面
// 特殊地在树中，SG[u] = (SG[v_1]+1) Xor (SG[v_2) Xor... Xor SG[v_N] ,v_N是以其子节点为根的子树所代表的局面


// 计算长度为n的两个序列a,b的循环卷积， 在序列末尾补0到长度为L=2^t， 满足L>2*n。
// 得到序列a',b'， 并且对b'进行翻转操作得到新的b'
// 原序列的循环卷积为:	c[k] = sumation{ a[i]*b[(i+k)%n] } k = 0,1,...,n-1
// a',b'的通过快速傅里叶变换得到的结果是 c'： 有c[k] = c'[k]+c'[L-n+k];
//
// 一阶的递推公式 f[n] = f[n-1]+f[n-2]（斐波那契) 可以写成矩阵的形式
// (f[n], f[n-1])' = { 1, 1; 1, 0 } * (f[n-1], f[n-2])'; 然后用快速幂算法。