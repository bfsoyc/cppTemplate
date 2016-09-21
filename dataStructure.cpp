


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