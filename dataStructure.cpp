// �ֵ���
struct Trie 
{ 
	int ch[maxnode][sigma_size]; 
	//maxnodeΪ�ֵ��������������һ��Ϊsigma_size*�ַ������ȣ���sigma_sizeΪ�ַ�����С��ÿ���������ӽ������ 
	int val[maxnode];//ÿ�������Ϣ 
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
	inline void maintain( int Idx ){//ά���ڵ���Ϣ,!!!����ȷ���ڵ㲻����Ҷ�ӽڵ� 
		s[Idx] = max( s[Lc[Idx]], s[Rc[Idx]] );
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


// �ڶ��ֲ��ҵ������У�ͨ���ڱ�������[l,r]��Ѱ��Ŀ��λ�ã������䳤��Ϊ2ʱ������l==mid�������
// Ϊ�˱�����ѭ������ѯ����ĸ���Ӧ����[mid+1,r]��[l,mid]֮һ�������ǵĲ��ҵ�Ŀ��λ��Ӧ�ø�Ϊԭλ����һλ�á�

// ��ȡ�������������У�������̰���㷨���ڹ̶�һ���˵������£��������ñ������䳤�ȵķ���Ѱ����һ���˵㣨�ȱ������ٶ��֣�

//������������Сֵ�Ǹ��ܾ�������⡣
//��������ά�����ǵ�ǰ�Կ�����Ϊ��Сֵ������
//���ǿ�����չΪ��һ�����Ӷ��е���һ�α�ʾ��ǰ�����£��ұ���һ��������С��ͬ����˼·�����յ�����������һ���µĵ������У������ٵ���ʱ���㲻�ö������������ˡ���������������ͬʱά���Կ�����Ϊ��Сֵ������
//�����ǵ����ģ��Աȶ���1�ĵ�����Ԫ�أ�������ڣ������2�Ķ���Ԫ�أ�������ڣ���õ���Сֵ��