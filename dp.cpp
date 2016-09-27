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

	int len = m;
	bool cond1 = true; // 记录A[m]A[m-1]..A[m-i+1]这路径是否符合条件一：这里是不含18
	bool cond2 = false;
	int preDigit = 0;
	int prefixMOD = 0;
	int offset = 0;	// 用于处理 lessThan 情况
	// some initialization 
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
		// 维护前缀条件
		if( preDigit==1 && A[m-i+1]==8 )
			cond1 = false;
		if( A[m-i+1]==2 || A[m-i+1]==3 || A[m-i+1]==5 )
			cond2 = true;
		preDigit = A[m-i+1];
		prefixMOD = (prefixMOD*10+A[m-i+1])%7;
	}
	return dp1[m][0]+dp2[m][0];
}

// n 位 m 进制 状态压缩
// 状态数小，转移方程不规则，每次转移步骤计数加一，问题求取最少步骤数的，可考虑BFS
inline void stateDecode( int s, int& n, int* state ){
	int m = n;// m == n
	for( int i(0); i < n ; i++ ){
		state[i] = s%m;
		s /= m;
	}
}
inline int stateEncode( int &n, int* state, int* _base){
	int s= 0;
	for( int i(0); i < n ; i++ ){
		s += _base[i]*state[i];
	}
	return s;
}