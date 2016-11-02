// 字典树
struct Trie 
{ 
	int ch[maxnode][sigma_size]; 
	//maxnode为字典树最大结点容量（一般为sigma_size*字符串长度），sigma_size为字符集大小（每个结点最多子结点数） 
	int val[maxnode];//每个结点信息 
	int sz;//目前树中的结点数 
	void init_Trie(){
		sz=1;memset(ch[0],0,sizeof(ch[0]));
		
	} 
	//初始化树。这里不写成构造函数便可以调用。 
	//int idx(char c){return c-'a';}这么简单的语句就不写成函数了 
	//插入字符串s，附加信息v(v不能等于0) 
	void insert(char* s,int v, int mask = -1) {
		int u=0,n;
		n = mask==-1 ? strlen(s):mask;

		for(int i=0;i<n;i++) {
			//int c=s[i]-'a'; 
			int c=s[i]-'0';
			if(!ch[u][c]){//结点不存在，则新建结点 
				memset(ch[sz],0,sizeof(ch[sz])); 
				val[sz]=0;//中间结点附加信息为0（表示这里不是终点） 
				ch[u][c]=sz++; 
			} 
			u=ch[u][c]; 
		} 
		if( val[u] == 0 ) // 这里对重复出现的单词进行处理
			val[u]=v; 
	} 
	bool _find(char *s){//未测试 
		int len=strlen(s),tmp; 
		int u=0,n=strlen(s); 
		for(int i=0;i<n;i++){ 
			int c=s[i]-'a'; 
			if(!ch[u][c]) 
				return false; 
			u=ch[u][c]; 
		} 
		return val[u]; 
	} 
}; 
// 线段树 (lazy mark)
struct SegmentTree{
	int from,to;
	int iL,iR;	
	int val,ret;
	void built( int from, int to ){
		this->from = from, this->to = to;
		blt( 1, from, to);
	}
	void update( int intervalLeft, int intervalRight, int value){
		this->iL = intervalLeft, this->iR = intervalRight, this->val = value;
		upd( 1, from, to);
	}
	int query( int intervalLeft, int intervalRight ){
		this->iL = intervalLeft, this->iR = intervalRight;
		ret = 0;
		qry( 1, from, to );
		return ret;
	}
protected:
	void blt( int idx, int l, int r){
		lazy[idx] = -1; // 初始化懒标记
		if( l == r ){ // leaf node
			s[idx] = a[l];
			return;
		}
		int mid = l+(r-l)/2;
		blt( idx*2, l, mid );
		blt( idx*2+1, mid+1, r);
		maintain(idx);
	}
	void maintain(int idx){
			s[idx] = s[idx*2] + s[idx*2+1];
	}
	void pushdown(int idx, int l, int r){ // 如果需要知道子节点的区间长度，需要利用l，r
		if( lazy[idx] == -1 ) return;
		lazy[idx*2] = lazy[idx*2+1] = lazy[idx];		
		// 传递标记后更新子结点需要维护的值，仅仅传递标记的话，结点自身的值肯定正确的。
		int mid = l+(r-l)/2;
		int len1 = mid-l+1, len2 = r-mid;
		s[idx*2] = len1*lazy[idx*2], s[idx*2+1] = len2*lazy[idx*2+1]; 
		lazy[idx] = -1; // -1 表示消除懒标记, 懒标记不为-1的结点，对应的区间所维护的值应该是正确的
	}
	void upd( int idx, int l, int r ){
		if( iL <= l && iR >= r ){
			lazy[idx] = val;
			s[idx] = val*(r-l+1); // 懒标记的更新一定伴随维护值的更新
			return;
		}
		pushdown(idx,l,r); // 懒标记的下传递
		int mid = l+(r-l)/2;
		if( iL <= mid )	upd( idx*2,l,mid);
		if( iR > mid ) upd( idx*2+1, mid+1, r);
		maintain(idx);
	}
	void qry( int idx, int l, int r ){
		if( iL <= l && iR >= r ){
			ret += s[idx];
			return;
		}
		pushdown(idx,l,r); // 懒标记的下传递
		int mid = l+(r-l)/2;
		if( iL <= mid )	qry( idx*2,l,mid);
		if( iR > mid ) qry( idx*2+1, mid+1, r);		
	}
};
// 对于区间修改的问题中，通常使用懒标记，不同类型的区间修改使用不同懒标记，并且这些懒标记具有不同
// 优先级，有两个地方会修改懒标记，区间修改与pushdown操作，均要对次优先级的标记进行特殊处理

// f(i,j)为对数组区间[i,j]内的数的操作，若要求所有f(i,j)的和，则分治是很好的思路，线段树是很适合的数据结构

// persistent segment tree
struct PSegmentTree{
	int from,to;
	int NEXT_FREE_INDEX;
	int iL, iR; 
	int val;
	int ret;

	int update( int intervalLeft, int intervalRight, int value , int versionRootIdx){
		this->iL = intervalLeft, this->iR = intervalRight, this->val = value;
		return upd( versionRootIdx, from, to );
	}
	int query( int intervalLeft, int intervalRight, int versionRootIdx){
		this->iL = intervalLeft, this->iR = intervalRight;
		ret = -1; // initialization
		qry( versionRootIdx, from, to );
		return ret;
	}
	int built( int from, int to ){
		this->from = from, this->to = to;
		NEXT_FREE_INDEX = 0; // initialization
		return blt( from,to );
	}
protected:
	int upd( int idx, int l, int r ){
		int Idx = NEXT_FREE_INDEX++; // index of the node in new version of segment tree
		s[Idx] = s[idx]; // new node always should be initialized with the value of the previous version
		Lc[Idx] = Lc[idx], Rc[Idx] = Rc[idx]; // as well as the pointer

		if( iL <= l && iR >= r ){ // this node is completely covered by the interval
			s[Idx] = max( s[idx], val );
			return Idx;
		}		
		int mid = l+(r-l)/2;
		if( iL <= mid ) 
			Lc[Idx] = upd( Lc[idx], l, mid );
		if( iR > mid )
			Rc[Idx] = upd( Rc[idx], mid+1, r);
		return Idx;
	}
	void qry( int idx, int l, int r ){
		ret = max( ret, s[idx] ); // update answer;
		if( iL <= l && iR >= r )
			return;

		int mid = l+(r-l)/2;
		if( iL <= mid )
			qry( Lc[idx], l, mid );
		if( iR > mid )
			qry( Rc[idx], mid+1, r);
	}
	int blt( int l, int r ){
		int Idx = NEXT_FREE_INDEX++;
		if( l == r ){ // leaf node
			s[Idx] = 1;
			return Idx;
		}
		int mid = l+(r-l)/2;
		Lc[Idx] = blt( l, mid );
		Rc[Idx] = blt( mid+1, r);
		maintain(Idx); // if s[Idx] = f( s[Lc[Idx]], s[Rc[Idx]] ) 
		return Idx;
	}
	inline void maintain( int Idx ){//维护节点信息,!!!这里确保节点不能是叶子节点 
		s[Idx] = max( s[Lc[Idx]], s[Rc[Idx]] );
	} 
};


// Range minimun query, dp[i][j]:以位置i开始，长度为2^j的区间的最小值所在的位置
// ST sparse table 的方法
void RMQ_init( int* A, int n){ // initialize on A[0...n-1]
	for( int i = 0 ; i < n ; i++ ) dp[i][0] = i; // A[i]
	for( int j = 1; (1<<j) <= n ; j++ )
		for( int i = 0 ; i + (1<<j) -1 < n ; i++ )
			dp[i][j] = A[ dp[i][j-1] ] < A[ dp[i+(1<<(j-1))][j-1] ] ? dp[i][j-1]:dp[i+(1<<(j-1))][j-1];
}
int RMQ( int L, int R, int *A ){
	int k = 0;
	while ( (1<<(k+1)) <= R-L+1 ) k++ ; // 如果2^(k+1) <= R-L+1, 那么k可以加1
	return A[ dp[L][k] ] < A[ dp[R-(1<<k)+1][k] ] ? dp[L][k]:dp[R-(1<<k)+1][k];
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


// 在二分查找的问题中，通常在闭区间内[l,r]内寻找目标位置，当区间长度为2时，出现l==mid的情况。
// 为了避免死循环，查询区间的更新应该是[mid+1,r]与[l,mid]之一。而我们的查找的目标位置应该改为原位置下一位置。

// 求取最大区间的问题中，若存在贪心算法，在固定一个端点的情况下，可以利用倍增区间长度的方法寻找另一个端点（先倍增，再二分）

//单调队列求最小值是个很经典的问题。
//单调队列维护的是当前仍可能作为最小值的数。
//我们可以拓展为：一个数从队列弹出一次表示当前窗口下，右边有一个数比他小，同样的思路，将刚弹出的数插入一个新的单调队列，当其再弹出时，便不用对其再做考虑了。所以这两个队列同时维护仍可能作为次小值的数。
//由于是单调的，对比队列1的第两个元素（如果存在）与队列2的队首元素（如果存在）便得到次小值。