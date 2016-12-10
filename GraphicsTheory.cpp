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

//最小生成树 prim 算法，连通图上的所有 N 个点，并且使得连接的线段的总长最短，
// 复杂度 O(mlogm)， m是边数
int inTree[maxn]; //		标记结点是否在生成树内
int prim(int n){	
	int d(0);
	memset( inTree,0,sizeof(inTree));
	int s = 0; inTree[s] = 1; // 随便将一个结点i放入生成树,这里是第一个结点
	priority_queue<pii,vector<pii>,greater<pii> > q;
	for( int h = head[s]; h!=-1; h=edges[h].next ) // 更新各结点到生成树的距离
		q.push( pii( edges[h].w, edges[h].v ) );
	
	for( int i(1); i < n ; i++ ){ // 每次找得1段，做 n-1次循环 
		while( inTree[q.top().second] ) q.pop(); // 若存在0权边，可能runtime error吗?
		int p = q.top().second;
		inTree[p] = 1; d += q.top().first; q.pop();
		for( int h = head[p]; h!=-1; h=edges[h].next ) // 更新各结点到生成树的距离
			q.push( pii(edges[h].w, edges[h].v) );
	}
	return d;
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

// LCA 离线查询 O(n+#query）
int root[maxn],vis[maxn],lca[maxn*2];	// 每天查询也以边的形式记录(双向),其结果存于lca中，别用vector<vector<int>>
struct Tarjan{
	void process(int n, int rootIdx){// n: 最大结点编号加1
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
			// 递归调用
			DFS( v );					
			// 将u与以其子节点v为根的子树合并
			root[v] = u;
		}
		// 处理与u相关的查询
		for( int h = queriesHead[u] ; h!=-1 ; h = queries[h].next )
			if( vis[queries[h].v] )
				lca[h] = lca[h^1] = find( queries[h].v ) ;
	}
};

// 并查集 
//带路径压缩的找根结点函数 
int find(int x){return root[x]==x?x:root[x]=find(root[x]);} 

// 2-sat(2-satisfiability)
// 该类问题抽象为：有一系列布尔型变量X={x1,x2,...,xi},其中某些变量对存在约束关系如xi!=xj为真，问是否存在满足约束的合法解。
struct TwoSAT{ 
	int n; 
	vector<int> G[maxn*2]; 
	bool mark[maxn*2]; 
	int S[maxn*2],c; //记录一条dfs进行的路径，用于回溯
 
	bool dfs( int x ){ 
	if( mark[x^1] ) return false;//  根据构建的边推导，出现矛盾则返回false 
	if( mark[x] ) return true;  //  算是剪枝 
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
 
	//  增加条件 ( x==xval or y==yval) == true 
	//  xval,yval 为 0（定义为真）或 1 
	//  若果x不等于y (x = !y),  则相当于增加两个条件( x==0 or y==0)==true 和 ( x==1 or y==1 )==true 
	void add_clause( int x, int xval, int y, int yval ){ 
		x = x*2 + xval; // 每个结点x拆分为两个结点 x*2 与 x*2+1
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

// 二分图判断并且标记 color[i] == 1 或 2分别表示两个子图
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


// 二分图的最大匹配 匈牙利算法 
vector<int> G[maxn];
int used[maxn],mat[maxn]; // mat[i] 表示左子图中与右子图结点i匹配的结点编号
bool hungery(int u){ // mat初始化1次， used在每次调用hungery前（不包括递归调用）初始化
	for(int i=0;i < G[u].size();i++){ 
		int v = G[u][i]; // 链表
		//c 是其中一个与 u(a)相连的右子图里的点（即他们右子图的 v 与左子图的 u 存在边） 
		if(!used[v]){//对于每个 a 而言，判断 v 是否访问过，属于dfs剪枝
			used[v] = 1;			
		//2 种情况任一成立则总匹配数加一，就是 main函数里的 sum++
			if(mat[v]==-1||hungery(mat[v])){ //这两个条件不能交换位置
				//1. v 未被匹配， 直接将u与v匹配
				//2. 左子图中的mat[v]找到右子图中v以外的匹配点k,即有更新mat[k]=mat[v], mat[v] = u
				mat[v]=u; 
				return true; 
			}  
		} 
	} 
	return false; 
} 

// 二分图的匹配问题
// 1. 最小点覆盖的点数(用最少的点覆盖所有的边) = 二分图最大匹配（匹配对的数目）
//		直观的理解是，最大匹配下，任何边都被覆盖了，而且去掉任何一个匹配都导致未被覆盖的边
// 2. 最大独立集的点数(选出一个最大子集，使得子集内的点都无边） = 总点数 - 二分图最大匹配

/* 树上的问题
	最长路径： 点集的直径用其两个端点维护，方便两个点集的合并。

*/