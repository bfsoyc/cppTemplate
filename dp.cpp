// 0-1�����Ķ������Ż�
void multItemsOpt( vector<int>& w, vector<int>& p, int W, int P, int num ){
	// ��num����Ʒת��Ϊ O(num) ����������Ʒ��
	int k(0);
	while( (1<<(k+1)) <= num ){ //��ȷ��W ����1Ŷ
		w.push_back( (1<<k)*W );
		p.push_back( (1<<k)*P );
		k++;
	}
	w.push_back( (num-(1<<k)+1)*W );	
	p.push_back( (num-(1<<k)+1)*P );
}