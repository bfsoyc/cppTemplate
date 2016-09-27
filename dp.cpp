#include<cstdio>
#include<isostream>
using namespace std;
typedef long long LL;

// 0-1�����Ķ������Ż�
void multItemsOpt( vector<int>& w, vector<int>& p, int W, int P, int num ){
	// ��num����Ʒת��Ϊ O(num) ����������Ʒ��
	int k(0);
	while( (1<<(k+1)) <= num ){ //��ȷ��W ����1Ŷ
		w.push_back( (1<<k)*W );
		p.push_back( (1<<k)*P );
		k++;
	}
	w.push_back( (num-(1<<k)+1)*W );	
	p.push_back( (num-(1<<k)+1)*P );
}

// ��λDP
LL digitDP( LL n ){ // �����1����n-1���ļ���
	// ���øú���ǰ�����ʼ������dp״̬����
	int m(0);
	while( n > 0 ){
		A[++m] = n%10;
		n /= 10;
	}
	// A[1]:���λ A[m]:���λ

	int len = m;
	bool cond1 = true; // ��¼A[m]A[m-1]..A[m-i+1]��·���Ƿ��������һ�������ǲ���18
	bool cond2 = false;
	int preDigit = 0;
	int prefixMOD = 0;
	int offset = 0;	// ���ڴ��� lessThan ���
	// some initialization 
	dp4[0][0] = 0;
	for( int i = 1 ; i <= len ; i++ ){ 
		// ÿ��ѭ������ǰm-iλȡ"A[m]A[m-1]..A[i-1]"����iλȡС��A[i]��iλ���ж�Ӧ�ļ���
		// �ɼ�����ѭ������������"A[m]A[m-1]...A[2]X"(��A[1]+1�����ļ�����cond1 �� preDigit ���ڸ��������
		// ö�ٵ�iλ
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
		// ά��ǰ׺����
		if( preDigit==1 && A[m-i+1]==8 )
			cond1 = false;
		if( A[m-i+1]==2 || A[m-i+1]==3 || A[m-i+1]==5 )
			cond2 = true;
		preDigit = A[m-i+1];
		prefixMOD = (prefixMOD*10+A[m-i+1])%7;
	}
	return dp1[m][0]+dp2[m][0];
}

// n λ m ���� ״̬ѹ��
// ״̬��С��ת�Ʒ��̲�����ÿ��ת�Ʋ��������һ��������ȡ���ٲ������ģ��ɿ���BFS
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