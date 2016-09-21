


// 字典树
struct Trie 
{ 
	int ch[maxnode][sigma_size]; 
	//maxnode为字典树最大结点容量（一般为sigma_size*字符串长度），sigma_size为字符集大小（每个结点最多子结点数） 
	int val[maxnode];//每个结点信息 
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