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