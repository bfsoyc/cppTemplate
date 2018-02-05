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

// palindromic tree
// 回文树 O(n*sigma)， sigma是字符集大小（通常为26),一种状态机类数据结构，每种状态表示一种回文串，总的状态数O(n)
char S[maxn];
int len[maxn], trans[maxn][26], slink[maxn], NEXT_FREE_IDX;
// len[i]: 状态i表示的回文串长度， trans: 状态转移指针， slink[maxn]: suffix-link 失配指针（fail)
int new_state(int length) {
	len[NEXT_FREE_IDX] = length;
	for (int i(0); i < 26; i++) trans[NEXT_FREE_IDX][i] = -1;
	return NEXT_FREE_IDX++;
}
int init() {
	NEXT_FREE_IDX = 0;
	int s1 = new_state(0);	//	中心为空串的回文串（长度为偶数）的开始状态
	int s2 = new_state(-1);	// 长度为奇数的回文串的开始状态,长度设置为-1是为了后面代码统一上的便利
	slink[s1] = s2;	// 等价于 slink[0] = 1;
	return s1;
}
int get_fail(int pos, int u) {
	while (S[pos - len[u] - 1] != S[pos]) u = slink[u];
	return u;
}
int add_char(int pos, int u) {	// 从状态u插入字符S[pos]
	int c = S[pos] - 'a';
	int mat = get_fail(pos, u);	// 沿着slink找到第一个match的状态。
	if (trans[mat][c] == -1) {	// 如果不存在该状态，则新建结点
		int v = new_state(len[mat] + 2);	// 长度加2
		trans[mat][c] = v;
		if (len[v] == 1) slink[v] = 0;	// 单字符构成的回文的前缀链指向应该是空串回文
		else slink[v] = trans[get_fail(pos, slink[mat])][c];	//总是存在的(除非mat==1,等价于当前len为1）
	}
	return trans[mat][c];
}
int getDistinctPalindromic() {	// 获得本质不同的回文串数目
								// S 必须从下表1开始
	S[0] = '#';
	int n = strlen(&S[1]);
	int last = init();
	for (int i(1); i <= n; i++) {
		last = add_char(i, last);
	}
	return NEXT_FREE_IDX - 2;
}

// 伸展树 splay tree
// 通常用于将序列转化为BST，实现快速的查找，特殊的伸展操作可以快速定位到区间，从而实现高效的区间修改操作。
// BST比较的键值其实是该元素在原序列中的位置，这个键值不用显式存储下来，通常维护的是另一个信息：以该节点为根的子树的大小(size)
struct Node {
	Node *ch[2];
	int val, sz, flip;	// val：值， sz: 大小size, flip: 区间翻转懒标记
	void maintain() {
		sz = ch[0]->sz + ch[1]->sz + 1;
	}
	void pushdown() {
		if (!flip) return;
		flip = 0;
		swap(ch[0], ch[1]);
		ch[0]->flip = !ch[0]->flip;
		ch[1]->flip = !ch[1]->flip;
	}
};
Node *null = new Node();

// d = 0 表示左旋， d = 1 表示右旋
// 左旋其实是提升o的右儿子，右旋是提升左儿子
void rotate(Node* &o, int d) {	// o 是指针的引用
	Node* k = o->ch[d ^ 1];
	o->ch[d ^ 1] = k->ch[d]; k->ch[d] = o;
	o->maintain(); k->maintain(); o = k;	//	一定先维护o，再维护k
}

// 伸展操作，将以o为根的子树中第k个字符旋转到根o的位置。
void splay(Node* &o, int k) {
	o->pushdown();
	int rk1 = o->ch[0]->sz + 1;	// 节点本身的序等于左子树的大小加1
	if (rk1 == k) return;	// 目标节点已经在根位置了

	int d1 = rk1 > k ? 1 : 0;	// d1 等于 1 表示目标节点在左子树
	if (!d1) k -= rk1;	//	如果需要在右子树内寻找，更新k
	Node* p = o->ch[d1 ^ 1];
	p->pushdown();
	int rk2 = p->ch[0]->sz + 1;
	if (rk2 != k) {	// o 的孩子依然不是目标节点，则递归地做，将目标旋转到p的孩子的位置上
		int d2 = rk2 > k ? 1 : 0;
		if (!d2) k -= rk2;
		splay(p->ch[d2 ^ 1], k);
		// 重点在这里， d1 和 d2 的异同指示 o , p 和 目标节点是否同线，同线则一定先提升p节点( 不是保证正确性，而是保证树尽可能平衡 ）
		if (d1 == d2) rotate(o, d1);
		else rotate(o->ch[d1 ^ 1], d2);
	}
	rotate(o, d1);	// 如果没有进入上面的条件语句，则表示目标节点就是o的孩子，直接进行依次旋转即可。
}

// 翻转区间 [L,R], 通常序列会插入起始符和终止符
void Flip(Node*& o, int L, int R) {	// 这里必须是指针的引用，否则会错
	splay(o, L);
	splay(o->ch[1], R - L + 2);
	o->ch[1]->ch[0]->flip ^= 1;
}

// 打印序列
void printSeq(Node* o) {
	if (o == null) return;
	o->pushdown();
	printSeq(o->ch[0]);
	if (o->val >= 0)	// 起始符和终止符用 -1 表示， 不打印
		printf("%c", 'a' + o->val);
	printSeq(o->ch[1]);
}

// 从序列构建bst
Node* build(int* seq, int st, int ed) {
	int mid = (st + ed) / 2;
	Node* o = new Node();
	o->val = seq[mid], o->ch[0] = null, o->ch[1] = null;
	if (st < mid)	o->ch[0] = build(seq, st, mid - 1);
	if (mid < ed)	o->ch[1] = build(seq, mid + 1, ed);
	o->maintain();
	return o;
}





// SAM: suffix auto machine
// 后缀自动机
/*
利用后缀自动机，可以在线性时间内计算所有长度子串出现次数最多的次数：
e.g. 所有长度为3的子串中出现次数最多的一个出现了多少次？
如果问题是求一个串中至少出现m次的串中最长的一个的长度的话，问题就很直观了。
现在要求至少在m个模板串中出现的子串中最长的一个的长度，需要对后缀自动机有更根本的理解。

后缀自动中一个节点代表一个状态，一个状态(st)是原串中一系列子串的集合，这些子串拥有相同的结束位置集合，记为endPoints(st)
这些子串互为后缀（短的一定是长的后缀），并且他们的长度是连续的，而这些子串本身记作subString(st)。对于一个状态，
|subString(st)|最大可达到O(n)级别，每个的长度也可能达到O(n), |endPoints(st)| 也可能O(n)级别，把他们完全记录下来是不现实的。
通常只用几个信息代表这个状态：
1、maxLen(st), minLen(st) 分别表示这个状态包含子串的最长与最短长度。
2、 trans(st, c ) 状态转移函数，状态st通过某字符c转移到的下一个的状态
3、 suffixLink( st ) 后缀链指针，在递增法构建SAM时尤为关键的辅助指针。
e.g."aabba"的5个后缀分别为：
aabba	abba	bba	ba	a
他们连续的一段一定属于同一个状态，假设划分为：
aabba	abba |	bba |	ba	a
假设上述对应状态分别为 x,y,z， 那么 suffixLink(x) = y , sufffixLink(y) = z ， suffixLink(z) = s ( s为SAM的起点）
增量构建SAM的算法中，每次在已经构建好的S[1...i]的SAM上构建S[1...i, i+1] 的SAM，相当于“插入”第i+1个字符c。
每次插入字符，首先肯定要添加一个节点z，endPoints(z) = { i+1 }
S[1,..,i+1] 肯定属于 subString(z)， 其某些连续的后缀也属于subString(z)，这步操作需要沿着suffixLink一直回溯到s节点（称为suffixPath）。
根据不同情况有不同的操作：
1、如果suffixPath上的节点v没有对应字符c的转移，则令trans(v,c) = z
2、如果suffixPath上的节点v有c的转移,假设为x，且maxLen(v)+1==maxLen(x), 那么suffixLink(z) = x
此时endPoints(x) =  endPoints(x) + {i+1}, 因为subString(v)包含 S[1,..i]的某个后缀，这个后缀加上c后被状态x所包含，x的
结束位置集合必然要包含c的位置的(即i+1)
3、如果suffixPath上的节点v有c的转移,假设为x，且maxLen(v)+1<maxLen(x), 即说明x包含一些从非v状态通过c转移过来的字符串，这些字符串
除去最后一个c后得到的串一定不是S[1,...,i]的后缀，这些串一定不是S[1,...,i+1]的后缀，所以让suffixLink(z)=x，是不适合的。
此时需要拆分x节点，得到新的节点y，就是将原来subString(x)中长度大于maxLen(v)+1和不大于的放到两个不同状态中（分别为x和y，旧x不存在了）
根据suffixLink的定义，suffixLink(x) = y 是肯定。
可以知道endPoints(x)和拆分前是一样的，而endPoints(y) = endPoints(x) + {i+1}
所有符合上述条件的v，令trans(v,c) = y,并且suffixLink(z) = y
***特别注意的是其实从节点z沿着suffixPath一直到s的所有节点p, 都应该有 i+1 属于endPoints(p)，如果要维护关于endPoints(st)的信息，需要在
添加字符后，从z沿着suffixLink回溯到s，维护整条suffixPath上的节点关于endPoints的信息。
对多串构建SAM的方法是，每处理一个新的串，都从s开始插入。


构建多串SAM后，任意一个状态st，其endPoints(st)集合中包含的位置出现在多少个模板中，则在subString(st)中的串就出现在多少个
模板中（记作cap(st))。所以如果cap(st)>=m , 则可以用 maxLen(st) 更新答案（或者最后遍历一次SAM的所有状态）。
cap(st)可以在增量构建SAM的过程中维护。在插入第i+1个字符c后,最后的状态是z,则需要从z沿着suffixLink回溯到s，维护整条suffixPath上的节点
的信息。我们用last(st)标记当前endPoints(st)中最大的一个元素归属的模板串，如果last(st)!=cur， 显然cap(st)是需要增加1的，并更新last(st)
否则不做修改，可以这样做是因为我们是按模板串的顺序来构建SAM的。
*/
int NEXT_FREE_IDX = 0;
const int maxn = 1e6 + 10;
const int sigma = 26;
int maxlen[2 * maxn], minlen[2 * maxn], trans[2 * maxn][sigma], slink[2 * maxn]; //每add一个字符最少增加1个，最多增加两个状态
int last[2 * maxn], cap[2 * maxn], cur;	// 多模板匹配问题
int new_state(int _maxlen, int _minlen, int* _trans, int _slink) {
	// 新建一个结点，并进行必要的初始化。
	maxlen[NEXT_FREE_IDX] = _maxlen;
	minlen[NEXT_FREE_IDX] = _minlen;
	for (int i(0); i < sigma; i++) {
		if (_trans == NULL)
			trans[NEXT_FREE_IDX][i] = -1;
		else
			trans[NEXT_FREE_IDX][i] = _trans[i];
	}
	slink[NEXT_FREE_IDX] = _slink;
	return NEXT_FREE_IDX++;
}
int add_src() { // 新建源点
	maxlen[0] = minlen[0] = 0; slink[0] = -1;
	for (int i(0); i<sigma; i++) trans[0][i] = -1;
	NEXT_FREE_IDX = 0;
	return NEXT_FREE_IDX++;
}
int add_char(char ch, int u) { // 新插入的字符ch在位置i
	int c = ch - 'a';
	int z = new_state(maxlen[u] + 1, -1, NULL, -1); // 新的状态只包含一个结束位置i
	int v = u;
	while (v != -1 && trans[v][c] == -1) {
		// 对于suffix-link 上所有没有对应字符ch的转移
		trans[v][c] = z;
		v = slink[v]; // 沿着suffix-link往回走
	}
	if (v == -1) {
		// 最简单的情况，整条链上都没有对应ch的转移
		minlen[z] = 1; // ch字符自身组成的子串
		slink[z] = 0;
		return z;
	}
	int x = trans[v][c];
	if (maxlen[v] + 1 == maxlen[x]) {
		// 不用拆分状态x的情况: 从v到x有对应ch的状态转移，但v代表的所有结束位置的后一位置不一定都是ch，故{x代表的结束位置}只是{v代表的结束位置+1}一个子集
		// x能代表更广泛（长度也就可以更长）的字符串，如果满足maxlen[v]+1 == maxlen[x],则v中的子串+ch就恰好与x中的子串一一对应
		// 此时 x 代表的结束位置就是{原来x代表的结束位置,位置i}
		minlen[z] = maxlen[x] + 1;
		slink[z] = x;
		//return z;
	}
	else {
		// 拆分x: x包含一连串连续的子串，将大于maxlen[y]+1的那些（仍然分配到x)和余下的(分配到y)分别拆分到x和y两个状态下
		// 那些能够通过ch转移到原来的x状态的所有状态中，某些要重新指向y，因为suffix-link和状态机的性质，很容易实现。
		// 同时 y 需要拷贝一份原来x状态的转移函数，见new_state();
		int y = new_state(maxlen[v] + 1, minlen[x]/*-1*/, trans[x], slink[x]);
		cap[y] = cap[x], last[y] = last[x];	//	 new_state() 不支持原始版本以外的信息拷贝。
											//slink[y] = slink[x]; // new_state中已赋值
		minlen[x] = maxlen[y] + 1; // = maxlen[v]+2 ; 拆分后，x包含的最短字符串和y包含的最长字符串需要更新
		slink[x] = y;
		minlen[z] = maxlen[y] + 1;
		slink[z] = y;

		int w = v;
		while (w != -1 && trans[w][c] == x) {
			trans[w][c] = y;
			w = slink[w];
		}
		//minlen[y] = maxlen[slink[y]]+1; //y的最短不就是原来x的最短了？ new_state中赋值
	}
	for (int p = z; p != -1; p = slink[p]) {
		if (last[p] != cur) {
			last[p] = cur;	cap[p]++;
		}
		else break;	// 这个优化至关重要？
	}
	return z;
}

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