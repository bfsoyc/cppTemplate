#include<algorithm> // for sort()
#include<vector>



// ��ɢ��a:  1 1 1000 1000 -> 2 2 4 4
int a[maxn];
bool cmp( int x, int y ){
	return a[x] < a[y];
}
void discretization(int n){
	vector<int> ID(n, 0);
	for( int i(0); i < n ; i++ ) ID[i] = i;
	sort(ID.begin(), ID.end(), cmp );
	for( int i(0); i < n ; ){
		int j(i);
		while( j < n && a[ID[j]]==a[ID[i]] ) j++;
		while( i<j ) a[ID[i++]] = j-1;
	}
}

// �����׺���� O(nlogn)
int cntA[maxn],cntB[maxn],rnk[maxn],A[maxn],B[maxn],tsa[maxn]; // ��������������
int ch[maxn],height[maxn],sa[maxn]; // ch:���ַ�����ת����õ����ַ����飬 sa:�����ĺ�׺���飬 height[i]: sa[i]��sa[i-1]������ĺ�׺�������ǰ׺
const int maxC = 1026;
void getSuffixArray(int n){
	// һ��ע���ʼ�� ch
    for (int i = 0; i < maxC; i ++) cntA[i] = 0; // ���ж�cntA�Ĳ���Ϊ����������� 
    for (int i = 1; i <= n; i ++) cntA[ch[i]] ++;
    for (int i = 1; i < maxC; i ++) cntA[i] += cntA[i - 1];
    for (int i = n; i; i --) sa[cntA[ch[i]] --] = i; // sa������������
    rnk[sa[1]] = 1;
    for (int i = 2; i <= n; i ++){
        rnk[sa[i]] = rnk[sa[i - 1]];
        if (ch[sa[i]] != ch[sa[i - 1]]) rnk[sa[i]] ++;
    }// �˴�rnk�õ��Ե���ĸ(����Ϊ1���Ӵ������� aabaaaab �õ� 11211112
    for (int l = 1; rnk[sa[n]] < n; l <<= 1){
		// ����������ģ�����ΪL���Ӵ����ɳ���Ϊ2*L���Ӵ�����˫�ؼ�������
        for (int i = 0; i <= n; i ++) cntA[i] = 0;
        for (int i = 0; i <= n; i ++) cntB[i] = 0;
        for (int i = 1; i <= n; i ++)
        {
            cntA[A[i] = rnk[i]] ++;
            cntB[B[i] = (i + l <= n) ? rnk[i + l] : 0] ++;
        }
        for (int i = 1; i <= n; i ++) cntB[i] += cntB[i - 1];
        for (int i = n; i; i --) tsa[cntB[B[i]] --] = i; // tsa ������󣨺���L�����Ӵ���������
        for (int i = 1; i <= n; i ++) cntA[i] += cntA[i - 1];
        for (int i = n; i; i --) sa[cntA[A[tsa[i]]] --] = tsa[i]; // ��tsa[n] ��tsa[i]���ӵڶ��ؼ������Ŀ�ʼ
        rnk[sa[1]] = 1;
        for (int i = 2; i <= n; i ++){
            rnk[sa[i]] = rnk[sa[i - 1]];
            if (A[sa[i]] != A[sa[i - 1]] || B[sa[i]] != B[sa[i - 1]]) rnk[sa[i]] ++;
        }
    }// �˴��õ���׺����suffix array ���� rank
    for (int i = 1, j = 0; i <= n; i ++){
		// ��������һ����ʵ suffix(k-1) ���� suffix(i-1)(rank[i-1])��ǰһλ����ôsuffix(k)�϶���suffix(i)ǰ,�������ǰ׺��H[rank[i-1]]-1
		// ��H[rank[i]] >= H[rank[i-1]-1];
        if (j) j --; // �൱�ڳ�ʼ��Ϊ height[rnk[i-1]]
        while (ch[i + j] == ch[sa[rnk[i] - 1] + j]) j ++;
		// ��ǰ����Height of suffix(i), ch[i+j]��suffix(i)�ĵ�j+1���ַ������ܿ��ܻ�û����H[rank[i]-1]�����������Ѿ�֪���������ǰ׺������j��,��������Ƚ�֪������ͬ�ַ�Ϊֹ
        height[rnk[i]] = j;
    }
} 

// KMP ƥ���㷨�� ���Ӷ�O(n), f Ϊʧ�亯��
void getFail( char* T, int* f){ // T Ϊģ�崮
	f[0] = 0, f[1] = 0;
	int m = strlen(T);
	for( int i(1); i < m ; i++ ){ // ����㵽f[m]
		int j=f[i];
		while( j && T[j]!=T[i] ) j = f[j];
		f[i+1] = T[i]==T[j]?j+1:0;
	}
}
int KMP( char* T, char* P, int* f ){
	int m = strlen(T), n = strlen(P) ;
	int j = 0, cnt(0); // ��ǰƥ�䵽��λ��
	getFail(T,f);
	for( int i(0); i < n ; i++ ){
		while( j&& T[j]!=P[i] ) j = f[j]; // ����ĸ��Ӷ��ڴˣ�ÿ��ѭ��j���ټ���1����j�������i�Σ�ÿ������1��
		if( T[j]==P[i] ) j++;
		if( j==m ){ cnt++;}// done
	}
	return cnt;
}

// Aho-Corasick automation
// AC�Զ�������ģ�崮ƥ��
int f[maxn], last[maxn];
struct AhoCorasickAutomata{
	Trie *trie; // �õ��ֵ���
	AhoCorasickAutomata( Trie* t ){ trie = t; }
	//void init( Trie& t){ this->trie = t; }
	void insert(char* s,int v ){ (*trie).insert(s,v); }
	int find( char* P ){ // invoke getFail() first!!!
		int n = strlen(P), cnt(0);
		int j = 0; // ��ǰƥ�䵽�Ľ��
		for( int i=0 ; i < n ; i++ ){
			int c = P[i]-'a';
			//while( j && !ch[j][c] ) j = f[j]; // ���ch[j]����һ���ַ��Ƕ�������P[i],��ʧ�����
			j = ch[j][c];
			if( val[j] ){//�ýڵ������ַ���������ģ�崮
				cnt++;
			}
			else if( val[last[j]] ){// �ýڵ������ַ�����ĳ����׺��ģ�崮
				cnt++;
			}
		}
		return cnt;
	}
	void getFail(){
		queue<int> q;
		f[0] = 0;
		// ��ʼ������
		for( int c = 0;  c < sigma_size; c++ ){
			int u = ch[0][c];
			if( u ){ f[u] = 0; q.push(u); last[u] = 0; }
		}
		// ��BFS�����ʧ�亯��
		while( !q.empty() ){
			int r = q.front(); q.pop();
			for( int c = 0; c < sigma_size; c++ ){
				int u = ch[r][c];
				// if( !u ) continue;
				if( !u ){ // С�Ż�
					ch[r][c] = ch[f[r]][c];
					continue;
				}
				q.push( u );
				int v = f[r];
				while (v && !ch[v][c]) v = f[v];
				f[u] = ch[v][c];
				last[u] = val[f[u]]? f[u]:last[f[u]];
			}
		}
	}
};

// ���㳤��Ϊn�������Ӵ���ϣ����ֵ�������� ���Ӷ�O(n!)
// ���� 1234 ��һ���Ӵ�����ǣ� 3-124  �����ֵ�����С��һ����1-2-3-4,����һ����4-3-2-1.
int mark[10];
vector<string> buff;
void comb( int n){
	vector<int> unMark;
	for( int i(0); i < n; i++ )if( !mark[i] ) unMark.push_back(i);
	if( unMark.empty() ){
		for( int i(0); i < buff.size(); i++ ){
			if(i) printf("-");
			printf("%s",buff[i].c_str());
		}
		printf("\n");
		return;
	}
	// ö�� unMark������Ϊm) �������Ӵ���2^m-1��,Ϊ�˼��ʹ�ö����Ƶ����뷽ʽʵ�֣��������
	vector<string> subStrs;
	int maxS = 1<<unMark.size();
	for( int s(1); s < maxS ; s++ ){ // empty string is not in consideration
		string str="";
		int bit = maxS>>1, j = 0;
		while( bit ){
			if( bit & s ) str.push_back(char('1'+unMark[j]));
			bit=bit>>1;	j++;
		}
		subStrs.push_back( str );
	}
	sort( subStrs.begin(), subStrs.end() ); // ���ֵ�����

	for( int i(0); i < subStrs.size(); i++ ){
		int sz = subStrs[i].size();
		for( int j(0); j < sz; j++ ){
			mark[ subStrs[i][j]-'1' ] = 1;
		}
		buff.push_back(subStrs[i]);
		comb(n);
		for( int j(0); j < sz; j++ ){
			mark[ subStrs[i][j]-'1' ] = 0;
		}
		buff.pop_back();
	}
}


// �ڹ����ַ������Ӵ������У������õ���׺���飬ͨ������ת����height��sa�����ϵ��������⣬���ֻ���ά����������
// ֵ���ر�С�ĵ������㣺1����ģ��Ĵ���ʵ����height��sa�����±�1��ʼ�ģ����������ݽṹ�洢ʱע��߽�
// 2���⣬�����Ҫ��������Сֵ��������ȡ��׺suffix(i)��suffix(j)���ǰ׺ʱ��ע����ҵ�������height[i+1,...,j]
// һ������������Ƕ�������L���󳤶�ΪL���Ӵ���������ظ�����k,���Ǽ�Ҫö��L����Ҫö����ʼλ��i�������������
// �ظ��Ӵ������ʣ���Ҫö�ٵ���ʼλ��i��������O(stringLength/L)�ģ�����sparse tableÿ�������ѯ���Ӷ�O(1),�ܵ�
// ���ӶȾ�Ȼֻ��O(nlogn)

// �ַ�������ĳ��ø�ʽ����  {%[*] [width] [{h | I | I64 | L}]type | ' ' | '\t' | '\n' | ��%����}
// %[aB'] ƥ��a��B��'��һԱ��̰����
// %[^a] ƥ���a�������ַ�������ֹͣ���룬̰����
// %4c ƥ��4�ֽڳ��ȵ��ַ���%[width]type��

// gets(char* s)����ȡ���ĵ���βʱ����NULL

// algorithm�� lower_boundʹ��:
// int pos = upper_bound(a,a+n,k)-a;