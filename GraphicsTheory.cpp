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
	// 该函数调用前需对done和dist进行初始化
	priority_queue<pii,vector<pii>,greater<pii> >q;//不用greater时省略，默认为大根堆 q.push(make_pair(dist[a],a));/起始点加入队列 
	q.push( pii(0,st) );
	dist[st] = 0;
	while(!q.empty()) 
	{ 
		pii u= q.top();q.pop(); 
		int x=u.second;//x获得u对应的结点编号 
		if(done[x]) continue;	
		//Dijkstra对每个结点只做一次，所以不用担心下面将某一结点重复加入队列 
		done[x]=1; 
		int j=G[x].size();//G是用vector做的链表 
		int t ,v; 
		for(int i=0;i<j;i++){ 
			v=G[x][i].v; 
			t=dist[x]+G[x][i].w; 
			//if(t<=600)//带深度限制，这里是600 
			if(dist[v]<0||dist[v]>t){ //这里将dist初始化为-1替代了INF 
				dist[v]=t;
				q.push(make_pair(dist[v],v)); 
			} 
		} 
	} 
}

// 最大流算法 限时吃紧的情况下请尽量压缩n的大小，测试样例未必极端
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
		memset( vis, 0 , sizeof(bool)*n ); // 注意bool型与int型大小不一样哦
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
	// 不使用使用dfs递归地实现找增广路(并且实现同时增广多条)，而是从源点开始一条一条地找
	// 找到后对整条路径进行增广操作。
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
		while( d[s] < n ){ // 任存在从s到t的路径
			if( u == t ){ // DFS到达汇点
				flow += Augmetn();
				u = s;
			}
			int ok = 0; // 阻塞标记
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
				// 被阻塞不一定是没有路可走，而是该路不满足d的条件，需要更新d
				for( int h = head[u] ; h != -1 ; h = edges[h].next ){
					Edge& e = edges[h];
					if(e.cap > e.flow) m = min(m, d[e.to] );
				}	// d[u] 严格递增
				if( --num[d[u]] == 0 ) break; // gap 优化
				num[d[u] = m+1 ] ++;
				cur[u] = head[u];
				if( u!=s ) u = edges[p[u]].from;
			}
		}
		return flow;
	}
};

//最大流问题 Dinic
struct Edge{ 
	int from, to, cap, flow; 
	Edge( int fr,int t, int c, int fl ):from(fr),to(t),cap(c),flow(fl){}; 
}; 
struct Dinic{ 
	int n,m,s,t;    //结点数，边数（包括反向狐），源点编号，汇点编号 
	vector <Edge> edges;  //边数。edges[e] 和 edges[e^1]互为反向狐 
	vector <int> G[maxn]; //邻接表，G[i][j] 表示结点i的第j条边在e数组中的序号 
	bool vis[maxn];    //BFS 使用 
	int d[maxn];    //从起点到i的距离 
	int cur[maxn];    //当前弧下标,加速用 
 
	void init(int n = 0){  
		edges.clear(); 
		m = 0; 
		for( int i = 0 ; i < n ; i++ ) //结点编号有可能等于n时，一定要用 "<=" ,最好调整传入的参数
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
 
	bool BFS(){  //建立层次图 
		memset( vis , 0 , sizeof(vis) ); 
		queue<int> Q; 
		Q.push(s);  d[s] = 0;  vis[s] = 1;  //初始化 
		while( !Q.empty() ){ 
			int x = Q.front(); Q.pop(); 
			for( int i = 0 ; i < G[x].size() ; i++ ){ 
				Edge& e = edges[G[x][i]]; 
				if( !vis[e.to] && e.cap > e.flow ){ //只考虑残留网络中的弧 
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
		for( int& i = cur[x]; i < G[x].size() ; i++ ){ //从上次考虑的弧开始 
			Edge& e = edges[G[x][i]]; 
			if( d[x] + 1 == d[e.to] &&   (f = DFS( e.to , min(a, e.cap-e.flow ))) > 0 ){//!!!赋值运算符优先级最低 
				e.flow += f;    
				edges[G[x][i]^1].flow -= f; 
				flow += f; 
				a -= f; 
				if( a==0 ) break; // 该弧没有残留了，无需继续深搜 最大流问题 
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

// dfs序，非递归
// DI: 入序， DO: 出序， L:层次
int fa[maxn],cur[maxn];
vector<int> getDFSOrder( int rootIdx, int *headIdx, int *DI, int *DO, int* L ){	
	vector<int> D;
	memset( DI, -1,sizeof(cur) ); 
	memcpy( cur, headIdx, sizeof(cur) );
	int u(rootIdx), v(rootIdx), h;
	fa[rootIdx] = -1;
	// 根节点
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
		cur[u] = edges[h].next;	// 删除该边，每条边只用一次		
		if( DI[v] == -1 ){ // first visit
			DI[v] = D.size();
			D.push_back( v );
			fa[v] = u;
			L[v] = L[fa[v]]+1;
		}
		else{//一条回溯的边，我们不处理
			v = u;
		}
	}
	return D;
}

// 最近公共祖先的在线查询算法 <O(N*logN,1)>
int E[maxn*2],I[maxn],O[maxn],L[maxn],EL[maxn*2];
struct LCA{
	// E 是DFS搜索的欧拉序,I记录节点第一次出现(进入）在E中的下标，O记录节点最后一次出现（离开）在E中的下标。	
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

// 并查集 
//带路径压缩的找根结点函数 
int find(int x){return root[x]==x?x:root[x]=find(root[x]);} 

/* 树上的问题
	最长路径： 点集的直径用其两个端点维护，方便两个点集的合并。

*/