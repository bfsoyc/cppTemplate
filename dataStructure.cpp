


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