// �ֵ���
int ch[maxnode][sigma_size];  //maxnodeΪ�ֵ��������������һ��Ϊsigma_size*�ַ������ȣ���sigma_sizeΪ�ַ�����С��ÿ���������ӽ������ 
int val[maxnode];//ÿ�������Ϣ 
struct Trie 
{ 
	
	int sz;//Ŀǰ���еĽ���� 
	void init_Trie(){
		sz=1;memset(ch[0],0,sizeof(ch[0]));
		
	} 
	//��ʼ���������ﲻд�ɹ��캯������Ե��á� 
	//int idx(char c){return c-'a';}��ô�򵥵����Ͳ�д�ɺ����� 
	//�����ַ���s��������Ϣv(v���ܵ���0) 
	void insert(char* s,int v, int mask = -1) {
		int u=0,n;
		n = mask==-1 ? strlen(s):mask;

		for(int i=0;i<n;i++) {
			//int c=s[i]-'a'; 
			int c=s[i]-'0';
			if(!ch[u][c]){//��㲻���ڣ����½���� 
				memset(ch[sz],0,sizeof(ch[sz])); 
				val[sz]=0;//�м��㸽����ϢΪ0����ʾ���ﲻ���յ㣩 
				ch[u][c]=sz++; 
			} 
			u=ch[u][c]; 
		} 
		if( val[u] == 0 ) // ������ظ����ֵĵ��ʽ��д���
			val[u]=v; 
	} 
	bool _find(char *s){//δ���� 
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

// �߶��� (lazy mark)
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
		lazy[idx] = -1; // ��ʼ�������
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
	void pushdown(int idx, int l, int r){ // �����Ҫ֪���ӽڵ�����䳤�ȣ���Ҫ����l��r
		if( lazy[idx] == -1 ) return;
		lazy[idx*2] = lazy[idx*2+1] = lazy[idx];		
		// ���ݱ�Ǻ�����ӽ����Ҫά����ֵ���������ݱ�ǵĻ�����������ֵ�϶���ȷ�ġ�
		int mid = l+(r-l)/2;
		int len1 = mid-l+1, len2 = r-mid;
		s[idx*2] = len1*lazy[idx*2], s[idx*2+1] = len2*lazy[idx*2+1]; 
		lazy[idx] = -1; // -1 ��ʾ���������, ����ǲ�Ϊ-1�Ľ�㣬��Ӧ��������ά����ֵӦ������ȷ��
	}
	void upd( int idx, int l, int r ){
		if( iL <= l && iR >= r ){
			lazy[idx] = val;
			s[idx] = val*(r-l+1); // ����ǵĸ���һ������ά��ֵ�ĸ���
			return;
		}
		pushdown(idx,l,r); // ����ǵ��´���
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
		pushdown(idx,l,r); // ����ǵ��´���
		int mid = l+(r-l)/2;
		if( iL <= mid )	qry( idx*2,l,mid);
		if( iR > mid ) qry( idx*2+1, mid+1, r);		
	}
};
// ���������޸ĵ������У�ͨ��ʹ������ǣ���ͬ���͵������޸�ʹ�ò�ͬ����ǣ�������Щ����Ǿ��в�ͬ
// ���ȼ����������ط����޸�����ǣ������޸���pushdown��������Ҫ�Դ����ȼ��ı�ǽ������⴦��

// f(i,j)Ϊ����������[i,j]�ڵ����Ĳ�������Ҫ������f(i,j)�ĺͣ�������Ǻܺõ�˼·���߶����Ǻ��ʺϵ����ݽṹ

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
	inline void maintain( int Idx ){//ά���ڵ���Ϣ,!!!����ȷ���ڵ㲻����Ҷ�ӽڵ� 
		s[Idx] = max( s[Lc[Idx]], s[Rc[Idx]] );
		sid[Idx] = s[Lc[Idx]] > s[Rc[Idx]]? sid[Lc[Idx]]:sid[Rc[Idx]];
	}   
};


// Range minimun query, dp[i][j]:��λ��i��ʼ������Ϊ2^j���������Сֵ���ڵ�λ��
// ST sparse table �ķ���
void RMQ_init( int* A, int n){ // initialize on A[0...n-1]
	for( int i = 0 ; i < n ; i++ ) dp[i][0] = i; // A[i]
	for( int j = 1; (1<<j) <= n ; j++ )
		for( int i = 0 ; i + (1<<j) -1 < n ; i++ )
			dp[i][j] = A[ dp[i][j-1] ] < A[ dp[i+(1<<(j-1))][j-1] ] ? dp[i][j-1]:dp[i+(1<<(j-1))][j-1];
}
int RMQ( int L, int R, int *A ){
	int k = 0;
	while ( (1<<(k+1)) <= R-L+1 ) k++ ; // ���2^(k+1) <= R-L+1, ��ôk���Լ�1
	return A[ dp[L][k] ] < A[ dp[R-(1<<k)+1][k] ] ? dp[L][k]:dp[R-(1<<k)+1][k];
}

// ��άƽ���ϵ��Ĳ���
const int NODE_CAPACITY = 10;
typedef int T;
struct QuadtreeNode{
	T left,right,up,down; // define the region
	vector<T> x,y;
	int childNode[4]; // ��������չ���������
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




// �ڶ��ֲ��ҵ������У�ͨ���ڱ�������[l,r]��Ѱ��Ŀ��λ�ã������䳤��Ϊ2ʱ������l==mid�������
// Ϊ�˱�����ѭ������ѯ����ĸ���Ӧ����[mid+1,r]��[l,mid]֮һ�������ǵĲ��ҵ�Ŀ��λ��Ӧ�ø�Ϊԭλ����һλ�á�

// ��ȡ�������������У�������̰���㷨���ڹ̶�һ���˵������£��������ñ������䳤�ȵķ���Ѱ����һ���˵㣨�ȱ������ٶ��֣�

//������������Сֵ�Ǹ��ܾ�������⡣
//��������ά�����ǵ�ǰ�Կ�����Ϊ��Сֵ������
//���ǿ�����չΪ��һ�����Ӷ��е���һ�α�ʾ��ǰ�����£��ұ���һ��������С��ͬ����˼·�����յ�����������һ���µĵ������У������ٵ���ʱ���㲻�ö������������ˡ���������������ͬʱά���Կ�����Ϊ��Сֵ������
//�����ǵ����ģ��Աȶ���1�ĵ�����Ԫ�أ�������ڣ������2�Ķ���Ԫ�أ�������ڣ���õ���Сֵ��

// �ڽṹ����������������� <, ע������λ���ùؼ���const���Σ����еڶ������ڵ���sort()����ʱ�Ǳ��룬ģ��涨ֻ��ʹ�ò��ı�����Ա�ıȽϺ���
// bool operator < ( const T x ) const{}

// ��ʱ����Ҫά��һ������a���������ĳ�����ԣ�����ͻ��ظ�Ԫ�ص�����ͣ���ǰ׺���͡�����B_1���Կ��������Ե�һ��Ԫ�ؿ�ʼ������ĺ���ɵ����顣
// ��ô�������B_1������������ȥa[1]�������޸ģ���õ������Եڶ���Ԫ�ؿ�ʼ������ĺͣ�������ƣ��ÿɳ��������߶�����ά��B_2һֱ��B_n��
// ���� ���ظ�Ԫ�ص������, �������ڲ������е�����a[i]������һ����С������
// �����߲�ѯ�У�����һ���ܺõķ���