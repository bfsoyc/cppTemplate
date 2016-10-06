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