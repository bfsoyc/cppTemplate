#include<cstdio>
#include<isostream>
using namespace std;
typedef long long LL;

// 0-1背包的二进制优化
void multItemsOpt( vector<int>& w, vector<int>& p, int W, int P, int num ){
	// 将num个物品转化为 O(num) 数量级的物品数
	int k(0);
	while( (1<<(k+1)) <= num ){ //请确保W 大于1哦
		w.push_back( (1<<k)*W );
		p.push_back( (1<<k)*P );
		k++;
	}
	w.push_back( (num-(1<<k)+1)*W );	
	p.push_back( (num-(1<<k)+1)*P );
}

// 数位DP
LL digitDP( LL n ){ // 计算从1到（n-1）的计数
	// 调用该函数前必须初始化所有dp状态数组
	int m(0);
	while( n > 0 ){
		A[++m] = n%10;
		n /= 10;
	}
	// A[1]:最低位 A[m]:最高位
	// some initialization 
	int len = m;
	bool cond1 = true; // 记录A[m]A[m-1]..A[m-i+1]这路径是否符合条件一：这里是不含18
	bool cond2 = false;
	int preDigit = 0;
	int prefixMOD = 0;
	dp4[0][0] = 0;
	for( int i = 1 ; i <= len ; i++ ){ 
		// 每个循环计算前m-i位取"A[m]A[m-1]..A[i-1]"，第i位取小于A[i]的i位数中对应的计数
		// 可见，该循环不包括对数"A[m]A[m-1]...A[2]X"(共A[1]+1个）的计数。cond1 与 preDigit 用于该特殊情况
		// 枚举第i位
		for( int j = 0; j <= 9; j++ ){
			if( j!=1 ){
				for( int k = 0 ; k < 7 ; k++ ){
					int resi = (k*10+j)%7;
					dp2[i][resi] += (dp1[i-1][k]+dp2[i-1][k]);				
				}
				if( cond1 && cond2 && j < A[m-i+1] )
						dp2[i][(prefixMOD*10+j)%7] ++;
			}

			if( j==1 ){
				for( int k = 0 ; k < 7 ; k++ ){
					int resi = (k*10+j)%7;
					dp1[i][resi] += (dp1[i-1][k]+dp2[i-1][k]);
					dp3[i][resi] += (dp3[i-1][k]+dp4[i-1][k]);					
				}
				if( cond1 && cond2 && j < A[m-i+1] )
					dp1[i][(prefixMOD*10+j)%7]++;
				if( cond1 && !cond2 && j < A[m-i+1] )
					dp3[i][(prefixMOD*10+j)%7]++;
			}
			if( j==2 || j== 3 || j== 5 ){
				for( int k = 0 ; k < 7 ; k++ ){
					int resi = (k*10+j)%7;
					dp2[i][resi] += (dp3[i-1][k]+dp4[i-1][k]);					
				}
				if( cond1 && cond2 && j < A[m-i+1] )
					dp2[i][(prefixMOD*10+j)%7]++;
			}
			if( j==0 || j==4 || j==6 || j==7 || j==9 || j==8 ){
				for( int k = 0 ; k < 7 ; k++ ){
					int resi = (k*10+j)%7;
					dp4[i][resi] += (dp3[i-1][k]+dp4[i-1][k]);					
				}
				if( cond1 && !cond2 && j < A[m-i+1] )
					dp4[i][(prefixMOD*10+j)%7]++;
			}
			if( j==8 ){
				for( int k = 0 ; k < 7 ; k++ ){
					int resi = (k*10+j)%7;
					dp2[i][resi] -= dp1[i-1][k];
					dp4[i][resi] -= dp3[i-1][k];
				}
				if( preDigit == 1 ){
					if( cond1 && cond2 && j < A[m-i+1] )	
						dp2[i][(prefixMOD*10+j)%7]--;
					if( cond1 && !cond2 && j < A[m-i+1] )
						dp4[i][(prefixMOD*10+j)%7]--;
				}
			}
			
		}
		// 维护A前缀 条件
		if( preDigit==1 && A[m-i+1]==8 )
			cond1 = false;
		if( A[m-i+1]==2 || A[m-i+1]==3 || A[m-i+1]==5 )
			cond2 = true;
		preDigit = A[m-i+1];
		prefixMOD = (prefixMOD*10+A[m-i+1])%7;
	}
	return dp1[m][0]+dp2[m][0];
}
// 路径计数是所有数位dp中基本的信息。
// 特殊地，如果状态转移与数的长度有关，那么维护一条前导0路径就十分必要了
// 在枚举第i(i非1)位为j（j非0)时，i-1个前导0加上这个j可以构成一条新路径，作为后面递推的基础。


// 非典型动态规划，一道树状dp的例程
void dfs( int u, int fa ){
	// 遍历子节点为根的子树最优解
	for( int h = head[u]; h!=-1; h=edges[h].next ){
		if( fa == edges[h].v ) continue;
		dfs( edges[h].v, u );
	}
	// merge two banch into one once
	vector<int> r( maxn*2+1, INT_MAX);
	vector<int> r1( maxn*2+1, INT_MAX);
	// 下面基于边的dfs，使用根节点进行初始化（看做花费为0的另一分支）
	r[0] = r[val[u]] = 0, r1[0] = r1[val[u]] = 0;
	for( int h = head[u]; h!=-1; h=edges[h].next ){ 
		int v = edges[h].v;
		if( fa == v ) continue;
		for( int i=r.size()-1; i >=0; i-- ){ // r是省略了一维的滚动数组
			if( r[i]==INT_MAX ) continue;
			// 值得注意的是，滚动数组默认了用dp[i-1]来初始化dp[i]（下标指隐藏维度），但是某些情况下是容易导致错误的
			// 比如不合法的状态需要额外标注的情况下。
			for( int k=1; k < r.size(); k++ ){			
				if( dp2[v][k]==INT_MAX ) continue;
				r1[i+k] = min( r1[i+k], r[i]+dp2[v][k]
				+ edges[h].w);
				r1[i+k] = min( r1[i+k], r1[i]+dp1[v][k]
				+ edges[h].w*2);
				if( dp1[v][k]==INT_MAX ) continue;
				r[i+k] = min( r[i+k], r[i]+dp1[v][k]
				+ edges[h].w*2);
			}		
		}
	}
	for( int i=0; i < r.size(); i++ ){
		dp1[u][i] = r[i];
		dp2[u][i] = r1[i];
	}
}

// 非典型动态，最长回文子串 O(n)
int dp[maxn*2+1];
int getLongestPralindrome( char* s, int n ){
	// preprocessing
	int m = 2*n+1;
	char* ss = new char[m];
	ss[0] = '#';
	for( int i(0); i < n; i++ ) ss[2*i+1]=s[i],ss[2*i+2]='#';
	// dp
	int j = 0, ans = 1;
	dp[0] = 0;
	for( int i(1); i < m ; i++ ){
		int k = j+dp[j]>=j ? min( dp[2*j-i], dp[j]+(j-i) ):0;
		while( i+k<m && i-k>=0 && ss[i+k]==ss[i-k] ) k++;
		ans = max( ans, dp[i] = k-1);
		if( i+dp[i]>j+dp[j] ) j = i;
	}
	return ans;
}

// n 位 m 进制 状态压缩
// 状态数小，转移方程不规则，每次转移步骤计数加一，问题求取最少步骤数的，可考虑BFS
inline void stateDecode( int s, int n, int* state ){
	int m = n;// m == n
	for( int i(0); i < n ; i++ ){
		state[i] = s%m;
		s /= m;
	}
}
inline int stateEncode( int n, int* state, int* _base){
	int s= 0;
	for( int i(0); i < n ; i++ ){
		s += _base[i]*state[i];
	}
	return s;
}