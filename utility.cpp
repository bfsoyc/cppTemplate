#include<algorithm> // for sort()
#include<vector>



// 离散化a:  1 1 1000 1000 -> 2 2 4 4
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

// 计算后缀数组 O(nlogn)
int cntA[maxn],cntB[maxn],rnk[maxn],A[maxn],B[maxn],tsa[maxn]; // 基数排序辅助数组
int ch[maxn],height[maxn],sa[maxn]; // ch:将字符数组转化后得到的字符数组， sa:排序后的后缀数组， height[i]: sa[i]与sa[i-1]所代表的后缀的最长公共前缀
const int maxC = 1026;
void getSuffixArray(int n){
	// 一定注意初始化 ch
    for (int i = 0; i < maxC; i ++) cntA[i] = 0; // 所有对cntA的操作为基数排序操作 
    for (int i = 1; i <= n; i ++) cntA[ch[i]] ++;
    for (int i = 1; i < maxC; i ++) cntA[i] += cntA[i - 1];
    for (int i = n; i; i --) sa[cntA[ch[i]] --] = i; // sa是排序后的索引
    rnk[sa[1]] = 1;
    for (int i = 2; i <= n; i ++){
        rnk[sa[i]] = rnk[sa[i - 1]];
        if (ch[sa[i]] != ch[sa[i - 1]]) rnk[sa[i]] ++;
    }// 此处rnk得到对单字母(长度为1的子串）的序， aabaaaab 得到 11211112
    for (int l = 1; rnk[sa[n]] < n; l <<= 1){
		// 将（已排序的）长度为L的子串构成长度为2*L的子串，用双关键字排序
        for (int i = 0; i <= n; i ++) cntA[i] = 0;
        for (int i = 0; i <= n; i ++) cntB[i] = 0;
        for (int i = 1; i <= n; i ++)
        {
            cntA[A[i] = rnk[i]] ++;
            cntB[B[i] = (i + l <= n) ? rnk[i + l] : 0] ++;
        }
        for (int i = 1; i <= n; i ++) cntB[i] += cntB[i - 1];
        for (int i = n; i; i --) tsa[cntB[B[i]] --] = i; // tsa 是排序后（后半段L长的子串）的索引
        for (int i = 1; i <= n; i ++) cntA[i] += cntA[i - 1];
        for (int i = n; i; i --) sa[cntA[A[tsa[i]]] --] = tsa[i]; // 从tsa[n] 到tsa[i]，从第二关键字最大的开始
        rnk[sa[1]] = 1;
        for (int i = 2; i <= n; i ++){
            rnk[sa[i]] = rnk[sa[i - 1]];
            if (A[sa[i]] != A[sa[i - 1]] || B[sa[i]] != B[sa[i - 1]]) rnk[sa[i]] ++;
        }
    }// 此处得到后缀数组suffix array 及其 rank
    for (int i = 1, j = 0; i <= n; i ++){
		// 基于这样一个事实 suffix(k-1) 排在 suffix(i-1)(rank[i-1])的前一位，那么suffix(k)肯定在suffix(i)前,他们最大前缀是H[rank[i-1]]-1
		// 故H[rank[i]] >= H[rank[i-1]-1];
        if (j) j --; // 相当于初始化为 height[rnk[i-1]]
        while (ch[i + j] == ch[sa[rnk[i] - 1] + j]) j ++;
		// 当前计算Height of suffix(i), ch[i+j]是suffix(i)的第j+1个字符，尽管可能还没计算H[rank[i]-1]，但是我们已经知道他们最大前缀至少是j了,继续往后比较知道不相同字符为止
        height[rnk[i]] = j;
    }
} 

// KMP 匹配算法， 复杂度O(n), f 为失配函数
void getFail( char* T, int* f){ // T 为模板串
	f[0] = 0, f[1] = 0;
	int m = strlen(T);
	for( int i(1); i < m ; i++ ){ // 会计算到f[m]
		int j=f[i];
		while( j && T[j]!=T[i] ) j = f[j];
		f[i+1] = T[i]==T[j]?j+1:0;
	}
}
int KMP( char* T, char* P, int* f ){
	int m = strlen(T), n = strlen(P) ;
	int j = 0, cnt(0); // 当前匹配到的位置
	getFail(T,f);
	for( int i(0); i < n ; i++ ){
		while( j&& T[j]!=P[i] ) j = f[j]; // 程序的复杂度在此，每次循环j最少减少1，而j最大增大i次，每次增大1。
		if( T[j]==P[i] ) j++;
		if( j==m ){ cnt++;}// done
	}
	return cnt;
}

// Aho-Corasick automation
// AC自动机：多模板串匹配
int f[maxn], last[maxn];
struct AhoCorasickAutomata{
	Trie *trie; // 用到字典树
	AhoCorasickAutomata( Trie* t ){ trie = t; }
	//void init( Trie& t){ this->trie = t; }
	void insert(char* s,int v ){ (*trie).insert(s,v); }
	int find( char* P ){ // invoke getFail() first!!!
		int n = strlen(P), cnt(0);
		int j = 0; // 当前匹配到的结点
		for( int i=0 ; i < n ; i++ ){
			int c = P[i]-'a';
			//while( j && !ch[j][c] ) j = f[j]; // 如果ch[j]的下一个字符们都不等于P[i],沿失配边走
			j = ch[j][c];
			if( val[j] ){//该节点代表的字符串本身是模板串
				cnt++;
			}
			else if( val[last[j]] ){// 该节点代表的字符串的某个后缀是模板串
				cnt++;
			}
		}
		return cnt;
	}
	void getFail(){
		queue<int> q;
		f[0] = 0;
		// 初始化队列
		for( int c = 0;  c < sigma_size; c++ ){
			int u = ch[0][c];
			if( u ){ f[u] = 0; q.push(u); last[u] = 0; }
		}
		// 按BFS序计算失配函数
		while( !q.empty() ){
			int r = q.front(); q.pop();
			for( int c = 0; c < sigma_size; c++ ){
				int u = ch[r][c];
				// if( !u ) continue;
				if( !u ){ // 小优化
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

// 计算长度为n的所有子串组合，按字典序输出。 复杂度O(n!)
// 例如 1234 的一个子串组合是： 3-124  其中字典序最小的一个是1-2-3-4,最大的一个是4-3-2-1.
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
	// 枚举 unMark（长度为m) 的所有子串共2^m-1个,为了简便使用二进制的掩码方式实现，最后排序。
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
	sort( subStrs.begin(), subStrs.end() ); // 按字典序排

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


// 在关于字符串的子串问题中，经常用到后缀数组，通常可以转化到height与sa数组上的搜索问题，二分或者维护单调队列
// 值得特别小心的有两点：1上述模板的代码实现中height与sa是以下标1开始的，用其他数据结构存储时注意边界
// 2此外，如果需要求区间最小值，例如求取后缀suffix(i)与suffix(j)的最长前缀时，注意查找的区间是height[i+1,...,j]
// 一个特殊的问题是对于所有L，求长度为L的子串的最长连续重复次数k,我们既要枚举L，又要枚举起始位置i，但巧妙地利用
// 重复子串的性质，需要枚举的起始位置i的数量是O(stringLength/L)的，利用sparse table每次区间查询复杂度O(1),总的
// 复杂度居然只有O(nlogn)

// 字符串处理的常用格式控制  {%[*] [width] [{h | I | I64 | L}]type | ' ' | '\t' | '\n' | 非%符号}
// %[aB'] 匹配a、B、'中一员，贪婪性
// %[^a] 匹配非a的任意字符，并且停止读入，贪婪性
// %4c 匹配4字节长度的字符（%[width]type）

// gets(char* s)当读取到文档结尾时返回NULL

// algorithm中 lower_bound使用:
// int pos = upper_bound(a,a+n,k)-a;