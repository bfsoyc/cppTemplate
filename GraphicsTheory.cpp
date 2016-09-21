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
	// �ú�������ǰ���done��dist���г�ʼ��
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