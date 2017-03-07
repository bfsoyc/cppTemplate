// 字典树
int ch[maxnode][sigma_size];  //maxnode为字典树最大结点容量（字符串长度的总和），sigma_size为字符集大小（每个结点最多子结点数） 
int val[maxnode];//每个结点信息 
struct Trie 
{ 
	
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
// with lazy mark
LL s[maxn*logn],lazy[maxn*logn];
int sid[maxn*logn],Lc[maxn*logn],Rc[maxn*logn];
struct PSegmentTree{
	int from,to;
	int NEXT_FREE_INDEX;
	int iL, iR; 
	LL val;
	LL ret;

	int update( int intervalLeft, int intervalRight, LL value , int versionRootIdx){
		this->iL = intervalLeft, this->iR = intervalRight, this->val = value;
		return upd( versionRootIdx, from, to );
	}
	//int query( int intervalLeft, int intervalRight, int versionRootIdx){
	//	this->iL = intervalLeft, this->iR = intervalRight;
	//	ret = 0; // initialization
	//	qry( versionRootIdx, from, to );
	//	return ret;
	//}
	int built( int from, int to ){
		this->from = from, this->to = to;
		NEXT_FREE_INDEX = 0; // initialization
		return blt( from,to );
	}
protected:
	void pushdown(int idx){
		if( lazy[idx]==0 ) return;
		
		int Idx = NEXT_FREE_INDEX++;
		int lc = Lc[idx];
		lazy[Idx] = lazy[lc]+lazy[idx], s[Idx] = s[lc]+lazy[idx], 
		sid[Idx] = sid[lc];
		Lc[Idx] = Lc[lc], Rc[Idx] = Rc[lc];
		Lc[idx] = Idx;
		Idx = NEXT_FREE_INDEX++;
		int rc = Rc[idx];
		lazy[Idx] = lazy[rc]+lazy[idx], s[Idx] = s[rc]+lazy[idx];
		sid[Idx] = sid[rc];
		Lc[Idx] = Lc[rc], Rc[Idx] = Rc[rc];
		Rc[idx] = Idx;

		lazy[idx] = 0;
	}
	int upd( int idx, int l, int r ){
		
		int Idx = NEXT_FREE_INDEX++; // index of the node in new version of segment tree
		if( NEXT_FREE_INDEX > maxNode ) a[-1] = 0;
		s[Idx] = s[idx]; // new node always should be initialized with the value of the previous version
		lazy[Idx] = lazy[idx], sid[Idx] = sid[idx];
		Lc[Idx] = Lc[idx], Rc[Idx] = Rc[idx]; // as well as the pointer

		if( iL <= l && iR >= r ){ // this node is completely covered by the interval
			// be very careful to add INF to lazy mark
			lazy[Idx] += val;
			s[Idx] += val;
			return Idx;
		}
		// typically we wouldn't change anything in the old version, so for pushdown operation, we create two temporary new node
		pushdown( Idx );
		int mid = l+(r-l)/2;
		if( iL <= mid ) 
			Lc[Idx] = upd( Lc[Idx], l, mid );
		if( iR > mid )
			Rc[Idx] = upd( Rc[Idx], mid+1, r);
		maintain(Idx);
		return Idx;
	}
	//void qry( int idx, int l, int r ){
	//	ret = max( ret, s[idx] ); // update answer;
	//	if( iL <= l && iR >= r )
	//		return;

	//	int mid = l+(r-l)/2;
	//	if( iL <= mid )
	//		qry( Lc[idx], l, mid );
	//	if( iR > mid )
	//		qry( Rc[idx], mid+1, r);
	//}
	int blt( int l, int r ){
		int Idx = NEXT_FREE_INDEX++;
		if( l == r ){ // leaf node
			s[Idx] = prefix[l];
			sid[Idx] = l;
			lazy[Idx] = 0;
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
		sid[Idx] = s[Lc[Idx]] > s[Rc[Idx]]? sid[Lc[Idx]]:sid[Rc[Idx]];
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

// 二维平面上的四叉树
const int NODE_CAPACITY = 10;
typedef int T;
struct QuadtreeNode{
	T left,right,up,down; // define the region
	vector<T> x,y;
	int childNode[4]; // 很容易拓展到多叉树啊
	QuadtreeNode(int l,int r, int u, int d ):left(l),right(r),up(u),down(d){
		for( int i(0); i < 4; i++ ) childNode[i] = -1;	
	}
	QuadtreeNode(){}
}nodes[maxn];
struct QuadTree{
	int NEXT_FREE_IDX,x,y,l,r,u,d;
	vector<int> X,Y;
	QuadTree(int left, int right, int up, int down){ 
		NEXT_FREE_IDX = 0; 
		nodes[NEXT_FREE_IDX++] = QuadtreeNode(left,right,up,down);
	}
	void divideNode(int id){
		QuadtreeNode &node = nodes[id];
		int midx = node.left + (node.right-node.left)/2;
		int midy = node.down + (node.up-node.down)/2;
		node.childNode[0] = NEXT_FREE_IDX;
		nodes[NEXT_FREE_IDX++] = QuadtreeNode(node.left,midx,node.up,midy);
		node.childNode[1] = NEXT_FREE_IDX;
		nodes[NEXT_FREE_IDX++] = QuadtreeNode(node.left,midx,midy,node.down);
		node.childNode[2] = NEXT_FREE_IDX;
		nodes[NEXT_FREE_IDX++] = QuadtreeNode(midx,node.right,node.up,midy);
		node.childNode[3] = NEXT_FREE_IDX;
		nodes[NEXT_FREE_IDX++] = QuadtreeNode(midx,node.right,midy,node.down);
	}
	void insert(int x, int y ){
		this->x = x, this->y = y;
		inst(0);
	}
	bool inst( int id){
		QuadtreeNode &node = nodes[id];
		if( x < node.left || x > node.right || y < node.down || y > node.up ) return false;
		if( node.x.size() < NODE_CAPACITY ){ // this node is not full
			node.x.push_back(x), node.y.push_back(y);
		}
		else{
			if( node.childNode[0]==-1 )
				divideNode(id);
			for( int i(0); i < 4; i ++ )
				if( inst(node.childNode[i] )) break;
		}
		return true;
	}
	void query(int l,int r, int u, int d ){
		this->l = l, this->r = r, this->u = u, this->d = d;
		X.clear(), Y.clear();
		qry(0);
	}
	void qry( int id ){
		QuadtreeNode &node = nodes[id];
		if( node.left > r || node.right < l || node.up < d || node.down > u ) return;
		for( int i(0); i < node.x.size(); i++ )
			X.push_back( node.x[i] ), Y.push_back( node.y[i] );
		if( node.childNode[0]==-1 ) return;
		for( int i(0); i < 4; i++ )
			qry( node.childNode[i] );
	}
};

// 树状数组(二叉索引树)
#define lowbit(x) (x&-x)
LL C[maxn],sz; // 辅助数组C初始化为0，n次add操作将需要维护的数组录入。 由于C存放的是前缀和，注意溢出。 ！！sz要初始化
LL sum( int x ){ // 查询 A[1]...A[x]的和,默认1-based的数组（若为0-based，先做一次位移)
	LL ret = 0;
	while( x > 0 ){
		ret += C[x]; 
		x-= lowbit(x);
	}
	return ret;
}
void add( int x, int d ){
	while( x <= sz ){
		C[x] += d;
		x += lowbit(x);
	}
}

// 两种特殊的问题下： 区间查询，点修改 或 点查询，区间修改
// 使用树状数组更便捷。对于点查询，区间修改问题:
// 构造差分数组 D[i] = A[i]-A[i-1]; => A[i] = (A[1]+...+A[i]) - (A[1]+...+A[i-1]) = D[1]+...+D[i]
// 而区间[l,r]修改(增量det)操作一般对应两次add操作: D[l]+det, D[r+1]-det;
// 其中 D[1] = A[1]; 


// 在二分查找的问题中，通常在闭区间内[l,r]内寻找目标位置，当区间长度为2时，出现l==mid的情况。
// 为了避免死循环，查询区间的更新应该是[mid+1,r]与[l,mid]之一。而我们的查找的目标位置应该改为原位置下一位置。

// 求取最大区间的问题中，若存在贪心算法，在固定一个端点的情况下，可以利用倍增区间长度的方法寻找另一个端点（先倍增，再二分）

//单调队列求最小值是个很经典的问题。
//单调队列维护的是当前仍可能作为最小值的数。
//我们可以拓展为：一个数从队列弹出一次表示当前窗口下，右边有一个数比他小，同样的思路，将刚弹出的数插入一个新的单调队列，当其再弹出时，便不用对其再做考虑了。所以这两个队列同时维护仍可能作为次小值的数。
//由于是单调的，对比队列1的第两个元素（如果存在）与队列2的队首元素（如果存在）便得到次小值。

// 在结构体中重载运算符，如 <, 注意两个位置用关键字const修饰，其中第二个对于调用sort()函数时是必须，模板规定只能使用不改变对象成员的比较函数
// bool operator < ( const T x ) const{}

// 有时候需要维护一个数组a所有区间的某个属性（区间和或不重复元素的区间和）。前缀“和”数组B_1可以看成所有以第一个元素开始的区间的和组成的数组。
// 那么这个数组B_1“所有数”减去a[1]（区间修改）便得到所有以第二个元素开始的区间的和，如此类推，用可持续化的线段树来维护B_2一直到B_n。
// 对于 不重复元素的区间和, 区别在于不是所有的数减a[i]，而是一个更小的区间
// 在离线查询中，这是一个很好的方案