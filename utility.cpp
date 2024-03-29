#include<algorithm> // for sort()
#include<vector>

// 链式前向星
struct Edge{
	int v,next;
}edges[maxn*2];
int head[maxn],numOfE(0);
void addEdge( int u, int v ){
	edges[numOfE].v = v, edges[numOfE].next = head[u], head[u] = numOfE++;
}

// 离散化a:  1 1 1000 1000 -> 1 1 3 3
int arr[maxn];
bool cmp(int a, int b) {
	return arr[a] < arr[b];
}
void discretization(int n, int *A) {
	vector<int> ID(n, 0);
	for (int i(0); i < n; i++) {
		ID[i] = i;	arr[i] = A[i];
	}
	sort(ID.begin(), ID.end(), cmp);
	for (int i(0); i < n; ) {
		int j(i);
		while (j < n && A[ID[j]] == A[ID[i]]) j++;
		while (i<j) A[ID[i++]] = j - 1;
	}
}

// 三分法求最小值
LL L = 0, R = 50;
while (L + 1 < R) {
	LL len = (R - L + 1) / 3;
	LL mid1 = L + len, mid2 = mid1 + len;
	if (f(mid1)>f(mid2)) { // f(mid1) >= f(mid2) 时，最小值一定不在 [L,mid1], 类似地 f(mid1) <= f(mid2) 时, 最小值一定不在[mid2,R]
		L = mid1;
	}
	else {
		if (R == mid2) R--;	// 对与长度为3的区间 mid2 == R
		else R = mid2;
	}
}
LL ans = min( f(L), f(R));// 额外判断 L 与 R

// 计算后缀数组 O(nlogn)
int cntA[maxn],cntB[maxn],rnk[maxn],A[maxn],B[maxn],tsa[maxn]; // 基数排序辅助数组
int ch[maxn],height[maxn],sa[maxn]; // ch:将字符数组(一般为0-based)转化后得到的数字数组（1-based)， sa:排序后的后缀数组， height[i]: sa[i]与sa[i-1]所代表的后缀的最长公共前缀
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
        for (int i = 0; i <= maxC; i ++) cntA[i] = 0;
        for (int i = 0; i <= maxC; i ++) cntB[i] = 0;
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
		// 故H[rank[i]] >= H[rank[i-1]]-1;
        if (j) j --; // 相当于初始化为 height[rnk[i-1]]-1
        while (ch[i + j] == ch[sa[rnk[i] - 1] + j]) j ++;
		// 当前计算Height of suffix(i), ch[i+j]是suffix(i)的第j+1个字符，尽管可能还没计算H[rank[i]-1]，但是我们已经知道他们最大前缀至少是j了,继续往后比较知道不相同字符为止
        height[rnk[i]] = j;
    }
} 

// SAM: suffix auto machine
// 后缀自动机
int NEXT_FREE_IDX = 0;
int maxlen[2*maxn+10], minlen[2*maxn+10], trans[2*maxn+10][26], slink[2*maxn+10]; //每add一个字符最少增加1个，最多增加两个状态
int edpts[2*maxn+10],indegree[2*maxn+10], containPrefix[2*maxn+10];
int new_state( int _maxlen, int _minlen, int* _trans, int _slink){
	// 新建一个结点，并进行必要的初始化。
	maxlen[NEXT_FREE_IDX] = _maxlen;
	minlen[NEXT_FREE_IDX] = _minlen;
	for( int i(0); i < 26; i++ ){
		if( _trans==NULL )
			trans[NEXT_FREE_IDX][i] = -1;
		else
			trans[NEXT_FREE_IDX][i] = _trans[i];
	}
	slink[NEXT_FREE_IDX] = _slink;
	return NEXT_FREE_IDX++;
}
void add_src(){ // 新建源点
	maxlen[0] = minlen[0] = 0; slink[0] = -1;
	for( int i(0); i<26; i++ ) trans[0][i] = -1;
	NEXT_FREE_IDX = 1;
}
int add_char( char ch, int u ){ // 新插入的字符ch在位置i
	int c = ch-'a';
	int z = new_state( maxlen[u]+1,-1,NULL,-1); // 新的状态只包含一个结束位置i
	containPrefix[z] = 1;
	int v = u;
	while( v!=-1 && trans[v][c]==-1 ){
		// 对于suffix-link 上所有没有对应字符ch的转移
		trans[v][c] = z;
		v = slink[v]; // 沿着suffix-link往回走
	}
	if( v==-1 ){
		// 最简单的情况，整条链上都没有对应ch的转移
		minlen[z] = 1; // ch字符自身组成的子串
		slink[z] = 0; indegree[0]++;
		return z;
	}
	int x = trans[v][c];
	if( maxlen[v]+1 == maxlen[x] ){
		// 不用拆分状态x的情况: 从v到x有对应ch的状态转移，但v代表的所有结束位置的后一位置不一定都是ch，故{x代表的结束位置}只是{v代表的结束位置+1}一个子集
		// x能代表更广泛（长度也就可以更长）的字符串，如果满足maxlen[v]+1 == maxlen[x],则v中的子串+ch就恰好与x中的子串一一对应
		// 此时 x 代表的结束位置就是{原来x代表的结束位置,位置i}
		minlen[z] = maxlen[x]+1;
		slink[z] = x; indegree[x]++;
		return z;
	}
	// 拆分x: x包含一连串连续的子串，将大于maxlen[y]+1的那些（仍然分配到x)和余下的(分配到y)分别拆分到x和y两个状态下
	// 那些能够通过ch转移到原来的x状态的所有状态中，某些要重新指向y，因为suffix-link和状态机的性质，很容易实现。
	// 同时 y 需要拷贝一份原来x状态的转移函数，见new_state();
	int y = new_state(maxlen[v]+1, minlen[x]/*-1*/, trans[x], slink[x]);  
	//slink[y] = slink[x]; // new_state中已赋值
	minlen[x] = maxlen[y]+1; // = maxlen[v]+2 ; 拆分后，x包含的最短字符串和y包含的最长字符串需要更新
	slink[x] = y;	indegree[y]++;
	minlen[z] = maxlen[y]+1;
	slink[z] = y;	indegree[y]++;
	int w = v;
	while( w!=-1 && trans[w][c]==x ){
		trans[w][c] = y;
		w = slink[w];
	}
	//minlen[y] = maxlen[slink[y]]+1; //y的最短不就是原来x的最短了？ new_state中赋值
	return z;
}	
void getEndPtCount(){ // 计算每个状态的结束位置计数，根据SAM的性质进行拓扑序递推。
	queue<int> q;
	for( int i(1); i < NEXT_FREE_IDX; i++ )if( !indegree[i] ){
		q.push(i);
	}
	while( !q.empty() ){
		int u = q.front(); q.pop();
		if( containPrefix[u] ) edpts[u]++; // 标记为绿色+1.
		edpts[ slink[u]] += edpts[u];
		if( !--indegree[slink[u]] ) q.push(slink[u]);
	}
}

// 根据SAM计算字符串的相关问题
// 1.计算模板串循环同构串在原串中出现的次数
int vis[2*maxn+10];
int calCyclicIsomorphism( char *T ){
	// 先在调用该函数前构建原串的SAM
    memset(vis,0,sizeof(vis));
	int ans = 0; // int type is sufficient
	int len;
	int n = len = strlen(T); // length of T
	for( int i(0); i < len-1; i++ ) T[len+i] = T[i]; // extended string of template, say T'
	len += n-1 ; // length of T' 
	int u = 0;
	int l = 0;
	for( int i(0); i < len; i++ ){
		int c = T[i]-'a';
		while( u!=0 && trans[u][c]==-1 ) // dp 从T[i-1]的(u,l)对计算T[i]的(u,l)对
			u = slink[u], l = maxlen[u];
		if( trans[u][c]!=-1 )
			u = trans[u][c], l++;
		else // u 已经是S(0)了并且没有转移。也就是说不存在P的子串是T'[0..i]的某个后缀
			l = 0;
		if( l > n ) // 找到串T'[i-l+1,...,i](实际就是T的某个循环子串)在SAM中的哪个状态。
			while( maxlen[slink[u]] >= n )
				u = slink[u], l = maxlen[u];
		if( l >= n && !vis[u] ) // 每一个状态只包含一个长度为n的子串？只统计一次
			vis[u] = 1, ans+=edpts[u];
	}
	return ans;
}

// 2.计算数字串中不同子串的和
// 多个串的处理是将所有串连在一起并且用特殊符号连接(例如:)
// 构建SAM的函数 new_state 和 add_char 略做修改以计算indegree
int indegree[2*maxn+10],valid_s[2*maxn+10];
LL sum[2*maxn+10];
int new_state( int _maxlen, int _minlen, int* _trans, int _slink){
	maxlen[NEXT_FREE_IDX] = _maxlen;
	minlen[NEXT_FREE_IDX] = _minlen;
	for( int i(0); i < 26; i++ ){
		if( _trans==NULL )
			trans[NEXT_FREE_IDX][i] = -1;
		else{
			trans[NEXT_FREE_IDX][i] = _trans[i];
			if( _trans[i]!=-1 ) indegree[_trans[i]]++;
		}
	}
	slink[NEXT_FREE_IDX] = _slink;
	return NEXT_FREE_IDX++;
}
int add_char( char ch, int u ){ // 新插入的字符ch在位置i
	int c = ch-'0'; // 改为'0'
	int z = new_state( maxlen[u]+1,-1,NULL,-1); // 新的状态只包含一个结束位置i
	int v = u;
	while( v!=-1 && trans[v][c]==-1 ){
		trans[v][c] = z; indegree[z]++;
		v = slink[v]; // 沿着suffix-link往回走
	}
	if( v==-1 ){
		minlen[z] = 1; // ch字符自身组成的子串
		slink[z] = 0;
		return z;
	}
	int x = trans[v][c];
	if( maxlen[v]+1 == maxlen[x] ){
		minlen[z] = maxlen[x]+1;
		slink[z] = x;
		return z;
	}
	int y = new_state(maxlen[v]+1, minlen[x], trans[x], slink[x]);  
	minlen[x] = maxlen[y]+1; 
	slink[x] = y;
	minlen[z] = maxlen[y]+1;
	slink[z] = y;
	int w = v;
	while( w!=-1 && trans[w][c]==x ){
		trans[w][c] = y; 
		indegree[x]--, indegree[y]++;
		w = slink[w];
	}
	return z;
}	
int substrSum(){ // 根据拓扑序递推不同子串的和。
	queue<int> q;
	q.push(0); valid_s[0] = 1;
	LL ret(0);
	while( !q.empty() ){
		int u = q.front(); q.pop();
		ret += sum[u];	ret %= MOD;
		for( int i(0); i < 26; i ++ ){
			int v = trans[u][i];
			if( v!=-1 && i < 10){
				sum[v] += sum[u]*10+i*valid_s[u],	sum[v] %= MOD;
				valid_s[v] += valid_s[u];
			}
			if( !--indegree[v] ) q.push(v);
		}		
	}
	return (int)ret;
}


// KMP 匹配算法， 复杂度O(n), f 为失配函数
// The failure function f[i] can be interpreted as the next index to compare if we fail to match index i for the template string.
// Another intuitive explaination for f[i] is the maximum length of prefix, which equals to the suffix of T[0:i-1].
// s.t. the prefix can't be the T[0:i-1] itself.
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

// 归并排序并统计逆序对
LL mergeSort(int *A, int *buf, int l, int r) {
	LL cnt(0);
	if (l == r) return cnt;
	int mid = (l + r) / 2;
	cnt += mergeSort(A, buf, l, mid);
	cnt += mergeSort(A, buf, mid + 1, r);
	int len1 = mid - l + 1, len2 = r - mid;
	int i = 0, j = 0;
	while (i < len1 && j < len2) {
		if (A[l + i] > A[mid + 1 + j])
			buf[l + i + j] = A[mid + 1 + j++], cnt += (len1 - i);
		else
			buf[l + i + j] = A[l + i++];
	}
	while (i < len1) buf[l + i + j] = A[l + i++];
	while (j < len2) buf[l + i + j] = A[mid + 1 + j++];
	for (int i(l); i <= r; i++) A[i] = buf[i];
	return cnt;
}

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

// Mo's algorithm: a general framework to deal with a bunch of offline queries in O(N*sqrt(N)).
// partition the entire interval into sqrt(N) blocks, of which the size is sqrt(N)
// group each query according by the belonging block of the start position of its corresponding interval.
// compute the result of each query group by group in sorted order. Note that the query in each group should also be in sorted order.
const int sqrtN = 200;
int block[maxn], QUERY_COUNT;	// block[i]: the index of block query(i,j) belongs to. 
pair<int, int> qry[maxn];	// queries
void addQuery(int l, int r) {
	block[QUERY_COUNT] = l / sqrtN;
	qry[QUERY_COUNT] = pair<int, int>(l, r);	QUERY_COUNT++;
}
bool cmp(int a, int b) {
	if (block[a] == block[b]) return qry[a].second < qry[b].second;
	return block[a] < block[b];
}
LL ans[maxn], val;
int ID[maxn], A[maxn], l, r;
void movePtr(int ptr, int d) {	//
	if (d == 1) {	// add A[ptr]		
		val += sum(A[ptr] - 1, C) + A[ptr];
		add(A[ptr], A[ptr], C);
		val += 1LL * (sum(maxn - 1, D) - sum(A[ptr] - 1, D)) * A[ptr];
		add(A[ptr], 1, D);
	}
	else {	// remove A[ptr]
		val -= sum(A[ptr] - 1, C) + A[ptr];
		add(A[ptr], -A[ptr], C);
		add(A[ptr], -1, D);
		val -= 1LL * (sum(maxn - 1, D) - sum(A[ptr] - 1, D)) * A[ptr];
	}
};
void solve() {
	//for BIT
	sz = maxn - 1;
	memset(C, 0, sizeof(C));
	memset(D, 0, sizeof(D));

	for (int i(0); i < QUERY_COUNT; i++)	ID[i] = i;
	sort(ID, ID + QUERY_COUNT, cmp);	// sort queries
	l = 1, r = 0, val = 0;
	for (int i(0); i < QUERY_COUNT; i++) {
		int id = ID[i];
		// adjust the left and right pionter 
		while (l < qry[id].first) movePtr(l++, -1);
		// 中间过程可能会出现 l > r 的时刻，你要保证的是增删操作满足交换律，先增后删和先删后增是一样的
		while (l > qry[id].first) movePtr(--l, +1);
		while (r < qry[id].second) movePtr(++r, 1);
		while (r > qry[id].second) movePtr(r--, -1);
		ans[id] = val;
	}
}

// bit set某些时候可以优化运算（集合取并或交）
// bitset<size> bs;
// bs.set() 全置位；	bs.reset() 全复位
// bs[i] = ture;	第i位置位

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
// 输出数字的前导0可以用 {%0[width]type}

// gets(char* s)当读取到文档结尾时返回NULL

// algorithm中 lower_bound使用:
// int pos = upper_bound(a,a+n,k)-a;

// sort的比较函数可以单独定义一个布尔型函数，set、map类的对比函数则需封装一下
/*
struct Cmp1 {	// 必须在一个类内重载()，而且必须时const member function
	bool operator() (const pair<int, int>& a, const pair<int, int>& b) const {
		if (a.first == b.first) return a.second < b.second;	// 只要pair不完全相同，就不相同，这个大小符号随意
		return a.first < b.first;
	}
};*/