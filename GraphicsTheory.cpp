#include<functional> // std:greater

typedef pair<int,int> pii; 

struct Edge{
	int v,w;
	Edge(){};
	Edge( int iv, int iw ):v(iv),w(iw){};
};
int done[maxn*maxn],dist[maxn*maxn];
vector<Edge> G[maxn*maxn];
void Dijkstra(int st){
	memset(done,0,sizeof(done));
	memset(dist,-1,sizeof(dist));
	priority_queue<pii,vector<pii>,greater<pii> >q;//����greaterʱʡ�ԣ�Ĭ��Ϊ����� q.push(make_pair(dist[a],a));/��ʼ�������� 
	q.push( pii(0,st) );
	dist[st] = 0;
	while(!q.empty()) 
	{ 
		pii u= q.top();q.pop(); 
		int x=u.second;//x���u��Ӧ�Ľ���� 
		if(done[x]) continue;	
		//Dijkstra��ÿ�����ֻ��һ�Σ����Բ��õ������潫ĳһ����ظ�������� 
		done[x]=1; 
		int j=G[x].size();//G����vector�������� 
		int t ,v; 
		for(int i=0;i<j;i++){ 
			v=G[x][i].v; 
			t=dist[x]+G[x][i].w; 
			//if(t<=600)//��������ƣ�������600 
			if(dist[v]<0||dist[v]>t){ //���ｫdist��ʼ��Ϊ-1�����INF 
				dist[v]=t;
				q.push(make_pair(dist[v],v)); 
			} 
		} 
	} 
}

//��С������ prim �㷨����ͨͼ�ϵ����� N ���㣬����ʹ�����ӵ��߶ε��ܳ���̣�
// ���Ӷ� O(mlogm)�� m�Ǳ���
int inTree[maxn]; //		��ǽ���Ƿ�����������
int prim(int n){	
	int d(0);
	memset( inTree,0,sizeof(inTree));
	int s = 0; inTree[s] = 1; // ��㽫һ�����i����������,�����ǵ�һ�����
	priority_queue<pii,vector<pii>,greater<pii> > q;
	for( int h = head[s]; h!=-1; h=edges[h].next ) // ���¸���㵽�������ľ���
		q.push( pii( edges[h].w, edges[h].v ) );
	
	for( int i(1); i < n ; i++ ){ // ÿ���ҵ�1�Σ��� n-1��ѭ�� 
		while( inTree[q.top().second] ) q.pop(); // ������0Ȩ�ߣ�����runtime error��?
		int p = q.top().second;
		inTree[p] = 1; d += q.top().first; q.pop();
		for( int h = head[p]; h!=-1; h=edges[h].next ) // ���¸���㵽�������ľ���
			q.push( pii(edges[h].w, edges[h].v) );
	}
	return d;
}

// ������㷨 ��ʱ�Խ���������뾡��ѹ��n�Ĵ�С����������δ�ؼ���
int head[maxn],p[maxn],d[maxn],cur[maxn],num[maxn];
bool vis[maxn];
struct Edge{
	int from, to, cap, flow, next;
}edges[maxn*maxn];
struct ISAP{
	int n,m,s,t;
	//vector<Edge> edges;
	
	void init(int n){
		this->n = n;
		//edges.clear();
		m = 0;
		memset( head, -1, sizeof(int)*n );
	};

	void addEdge( int from, int to, int cap){
		//Edge e1(from, to, cap, 0 );
		edges[m].from = from, edges[m].to = to, edges[m].cap = cap, edges[m].flow = 0, edges[m].next = head[from];
		//e1.next = head[from];
		head[from] = m++;
		//edges.push_back( e1 );

		//Edge e2(to, from, 0, 0 );
		edges[m].from = to, edges[m].to = from, edges[m].cap = 0, edges[m].flow = 0, edges[m].next = head[to];
		//e2.next = head[to];
		head[to] = m++;
		//edges.push_back( e2 );
	}

	void backwardBFS(){	
		memset( vis, 0 , sizeof(bool)*n ); // ע��bool����int�ʹ�С��һ��Ŷ
		queue<int> Q;
		Q.push(t); d[t] = 0; vis[t] =true;
		while( !Q.empty() ){
			int u = Q.front(); Q.pop();
			for( int h = head[u] ; h!=-1; h = edges[h].next ){
				Edge& e = edges[h^1];
				if( !vis[e.from] && e.cap > e.flow ){
					vis[e.from] = 1;
					d[e.from] = d[u] + 1;
					Q.push( e.from );
				}
			}
		}
		//return vis[s];
	}
	// ��ʹ��ʹ��dfs�ݹ��ʵ��������·(����ʵ��ͬʱ�������)�����Ǵ�Դ�㿪ʼһ��һ������
	// �ҵ��������·���������������
	int Augmetn(){
		int u = t, a = INT_MAX;
		while( u != s ){
			Edge &e = edges[p[u]];
			a = min( a, e.cap-e.flow );
			u = e.from;
		}
		u = t;
		while( u != s ){
			edges[p[u]].flow += a;
			edges[p[u]^1].flow -= a;
			u = edges[p[u]].from;
		}
		return a;
	}
	int Maxflow( int s, int t){
		this->s = s; this->t = t;
		int flow = 0;
		memset( d, 0, sizeof(int)*n );
		backwardBFS();
		memset( num, 0, sizeof(int)*n );
		for(int i(0) ; i < n ; i++ ) num[d[i]]++;
		int u = s;
		memcpy( cur, head, sizeof(int)*n );
		while( d[s] < n ){ // �δ��ڴ�s��t��·��
			if( u == t ){ // DFS������
				flow += Augmetn();
				u = s;
			}
			int ok = 0; // �������
			for( int h = cur[u] ; h!=-1 ; h = edges[h].next ){
				Edge& e = edges[h];
				if( e.cap > e.flow && d[u] == d[e.to] + 1 ){ // Advance
					ok = 1;
					p[e.to] = h;
					cur[u] = h;
					u = e.to;
					break;
				}
			}

			if( !ok ){
				int m = n-1;	// Retreat
				// ��������һ����û��·���ߣ����Ǹ�·������d����������Ҫ����d
				for( int h = head[u] ; h != -1 ; h = edges[h].next ){
					Edge& e = edges[h];
					if(e.cap > e.flow) m = min(m, d[e.to] );
				}	// d[u] �ϸ����
				if( --num[d[u]] == 0 ) break; // gap �Ż�
				num[d[u] = m+1 ] ++;
				cur[u] = head[u];
				if( u!=s ) u = edges[p[u]].from;
			}
		}
		return flow;
	}
};

//��������� Dinic
struct Edge{ 
	int from, to, cap, flow; 
	Edge( int fr,int t, int c, int fl ):from(fr),to(t),cap(c),flow(fl){}; 
}; 
struct Dinic{ 
	int n,m,s,t;    //��������������������������Դ���ţ������ 
	vector <Edge> edges;  //������edges[e] �� edges[e^1]��Ϊ����� 
	vector <int> G[maxn]; //�ڽӱ�G[i][j] ��ʾ���i�ĵ�j������e�����е���� 
	bool vis[maxn];    //BFS ʹ�� 
	int d[maxn];    //����㵽i�ľ��� 
	int cur[maxn];    //��ǰ���±�,������ 
 
	void init(int n = 0){  
		edges.clear(); 
		m = 0; 
		for( int i = 0 ; i < n ; i++ ) //������п��ܵ���nʱ��һ��Ҫ�� "<=" ,��õ�������Ĳ���
			G[i].clear(); 
	}; 
	void AddEdge( int from , int to , int cap ){ 
		Edge e1(from, to, cap, 0); 
		Edge e2(to, from, 0, 0); 
		edges.push_back( e1 ); 
		edges.push_back( e2 ); 
		m = edges.size(); 
		G[from].push_back( m-2 ); 
		G[to].push_back( m-1 ); 
	} 
 
	bool BFS(){  //�������ͼ 
		memset( vis , 0 , sizeof(vis) ); 
		queue<int> Q; 
		Q.push(s);  d[s] = 0;  vis[s] = 1;  //��ʼ�� 
		while( !Q.empty() ){ 
			int x = Q.front(); Q.pop(); 
			for( int i = 0 ; i < G[x].size() ; i++ ){ 
				Edge& e = edges[G[x][i]]; 
				if( !vis[e.to] && e.cap > e.flow ){ //ֻ���ǲ��������еĻ� 
					vis[e.to] = 1; 
					d[e.to] = d[x] + 1; 
					Q.push( e.to ); 
				} 
			} 
		} 
		return vis[t]; 
	} 
 
	int DFS( int x ,int a ){ //
		if( x==t || a == 0 ) return a ; 
		int flow = 0 ,f; 
		for( int& i = cur[x]; i < G[x].size() ; i++ ){ //���ϴο��ǵĻ���ʼ 
			Edge& e = edges[G[x][i]]; 
			if( d[x] + 1 == d[e.to] &&   (f = DFS( e.to , min(a, e.cap-e.flow ))) > 0 ){//!!!��ֵ��������ȼ���� 
				e.flow += f;    
				edges[G[x][i]^1].flow -= f; 
				flow += f; 
				a -= f; 
				if( a==0 ) break; // �û�û�в����ˣ������������ ��������� 
			}
		}
		return flow; 
	} 
	int Maxflow( int s ,int t ){ 
		this->s = s;  this->t = t; 
		int flow = 0; 
		while( BFS() ){ 
			memset( cur , 0 , sizeof( cur )); 
			flow += DFS( s ,INF ); 
		} 
		return flow; 
	} 	
};

// dfs�򣬷ǵݹ�
// DI: ���� DO: ���� L:���
int fa[maxn],cur[maxn];
vector<int> getDFSOrder( int rootIdx, int *headIdx, int *DI, int *DO, int* L ){	
	vector<int> D;
	memset( DI, -1,sizeof(cur) ); 
	memcpy( cur, headIdx, sizeof(cur) );
	int u(rootIdx), v(rootIdx), h;
	fa[rootIdx] = -1;
	// ���ڵ�
	DI[rootIdx]=D.size(),L[rootIdx]=1;
	D.push_back(rootIdx);
	while( u >= 0 ){
		u = v;
		DO[u] = D.size()-1;
		if( cur[u] == -1 ){
			v = fa[u];
			continue;
		}
		h = cur[u];
		v = edges[h].v;
		cur[u] = edges[h].next;	// ɾ���ñߣ�ÿ����ֻ��һ��		
		if( DI[v] == -1 ){ // first visit
			DI[v] = D.size();
			D.push_back( v );
			fa[v] = u;
			L[v] = L[fa[v]]+1;
		}
		else{//һ�����ݵıߣ����ǲ�����
			v = u;
		}
	}
	return D;
}

// ����������ȵ����߲�ѯ�㷨 <O(N*logN,1)>
int E[maxn*2],I[maxn],O[maxn],L[maxn],EL[maxn*2];
struct LCA{
	// E ��DFS������ŷ����,I��¼�ڵ��һ�γ���(���룩��E�е��±꣬O��¼�ڵ����һ�γ��֣��뿪����E�е��±ꡣ	
	int CUR_IDX;
	void DFS( int u , int level, int fa){
		// first visit insertion
		I[u] = CUR_IDX,O[u] = CUR_IDX,L[u] =  level;
		E[CUR_IDX++] = u;
		for( int i = head[u] ; i != -1 ; i = edges[i].next ){
			if( edges[i].v == fa ) continue;
			DFS( edges[i].v, level+1, u);
			// back track visit insertion
			O[u] = CUR_IDX;
			E[CUR_IDX++] = u;
		}
	}
	void init(int rootIdx){
		CUR_IDX = 0;
		DFS( rootIdx, 1, -1 );
		for( int i(0); i < CUR_IDX; i++ )
			EL[i] = L[E[i]];
		RMQ_init( EL, CUR_IDX );
	}
	int lca( int u, int v ){
		int L = I[u], R = I[v];
		if( L > R )
			L ^= R^= L ^= R;
		return E[ RMQ(L,R,EL) ];
	}
};

// LCA ���߲�ѯ O(n+#query��
int root[maxn],vis[maxn],lca[maxn*2];	// ÿ���ѯҲ�Աߵ���ʽ��¼(˫��),��������lca�У�����vector<vector<int>>
struct Tarjan{
	void process(int n, int rootIdx){// n: ������ż�1
		for( int i(0) ; i < n ; i++ )
			root[i] = i;
		memset( vis,0,sizeof(int)*n );
		DFS( rootIdx );
	}

	int find( int x ){ return root[x]==x? x:root[x]=find(root[x]);} 

	void DFS( int u ){
		vis[u] = 1;
		for( int i = head[u] ; i != -1 ; i = edges[i].next ){
			int v = edges[i].v;
			if( vis[v] ) continue;
			// first visit
			// �ݹ����
			DFS( v );					
			// ��u�������ӽڵ�vΪ���������ϲ�
			root[v] = u;
		}
		// ������u��صĲ�ѯ
		for( int h = queriesHead[u] ; h!=-1 ; h = queries[h].next )
			if( vis[queries[h].v] )
				lca[h] = lca[h^1] = find( queries[h].v ) ;
	}
};

// ���鼯 
//��·��ѹ�����Ҹ���㺯�� 
int find(int x){return root[x]==x?x:root[x]=find(root[x]);} 

// 2-sat(2-satisfiability)
// �����������Ϊ����һϵ�в����ͱ���X={x1,x2,...,xi},����ĳЩ�����Դ���Լ����ϵ��xi!=xjΪ�棬���Ƿ��������Լ���ĺϷ��⡣
struct TwoSAT{ 
	int n; 
	vector<int> G[maxn*2]; 
	bool mark[maxn*2]; 
	int S[maxn*2],c; //��¼һ��dfs���е�·�������ڻ���
 
	bool dfs( int x ){ 
	if( mark[x^1] ) return false;//  ���ݹ����ı��Ƶ�������ì���򷵻�false 
	if( mark[x] ) return true;  //  ���Ǽ�֦ 
		mark[x] = true; 
		S[c++] = x; 
		for( int i = 0 ;   i < G[x].size(); i++ )if( !dfs( G[x][i] ) ) return false; 
   
		return true;  
	} 
 
	void init( int n ){ // 0-based
		this->n = n; 
		for( int i = 0 ; i < n*2 ; i++ ) G[i].clear(); 
		memset( mark,0,sizeof( mark ) );//   
	} 
 
	//  �������� ( x==xval or y==yval) == true 
	//  xval,yval Ϊ 0������Ϊ�棩�� 1 
	//  ����x������y (x = !y),  ���൱��������������( x==0 or y==0)==true �� ( x==1 or y==1 )==true 
	void add_clause( int x, int xval, int y, int yval ){ 
		x = x*2 + xval; // ÿ�����x���Ϊ������� x*2 �� x*2+1
		y = y*2 + yval; 
		G[x^1].push_back( y ); 
		G[y^1].push_back( x ); 
	} 
 
	bool solve(){ 
		for( int i = 0 ; i < n*2 ; i+= 2 )if( !mark[i] && !mark[i+1] ){ 
			c = 0; 
			if( !dfs(i) ){ 
				while( c > 0 ) mark[S[--c]] = false; 
				if( !dfs(i+1) ) return false; 
			} 
		} 
		return true; 
	} 
}; 

// ����ͼ�жϲ��ұ�� color[i] == 1 �� 2�ֱ��ʾ������ͼ
int color[maxn];
bool solve( int n){
	for( int i(1); i <= n ; i++ )if( !color[i] ){
		color[i] = 1;
		queue<int> q; q.push(i);
		while( !q.empty() ){
			int u = q.front(); q.pop();
			for( int h = head[u]; h!=-1 ; h=edges[h].next ){
				int v = edges[h].v;
				if( color[u]==color[v] ) return false;
				if( color[v]==0 ){
					color[v] = color[u]==1?2:1;
					q.push( v );
				}
			}
		}
	}
	return true;
}


// ����ͼ�����ƥ�� �������㷨 
vector<int> G[maxn];
int used[maxn],mat[maxn]; // mat[i] ��ʾ����ͼ��������ͼ���iƥ��Ľ����
bool hungery(int u){ // mat��ʼ��1�Σ� used��ÿ�ε���hungeryǰ���������ݹ���ã���ʼ��
	for(int i=0;i < G[u].size();i++){ 
		int v = G[u][i]; // ����
		//c ������һ���� u(a)����������ͼ��ĵ㣨����������ͼ�� v ������ͼ�� u ���ڱߣ� 
		if(!used[v]){//����ÿ�� a ���ԣ��ж� v �Ƿ���ʹ�������dfs��֦
			used[v] = 1;			
		//2 �������һ��������ƥ������һ������ main������� sum++
			if(mat[v]==-1||hungery(mat[v])){ //�������������ܽ���λ��
				//1. v δ��ƥ�䣬 ֱ�ӽ�u��vƥ��
				//2. ����ͼ�е�mat[v]�ҵ�����ͼ��v�����ƥ���k,���и���mat[k]=mat[v], mat[v] = u
				mat[v]=u; 
				return true; 
			}  
		} 
	} 
	return false; 
} 

// ����ͼ��ƥ������
// 1. ��С�㸲�ǵĵ���(�����ٵĵ㸲�����еı�) = ����ͼ���ƥ�䣨ƥ��Ե���Ŀ��
//		ֱ�۵�����ǣ����ƥ���£��κα߶��������ˣ�����ȥ���κ�һ��ƥ�䶼����δ�����ǵı�
// 2. ���������ĵ���(ѡ��һ������Ӽ���ʹ���Ӽ��ڵĵ㶼�ޱߣ� = �ܵ��� - ����ͼ���ƥ��

/* ���ϵ�����
	�·���� �㼯��ֱ�����������˵�ά�������������㼯�ĺϲ���

*/