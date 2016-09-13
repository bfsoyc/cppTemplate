// 0-1背包的二进制优化
void multItemsOpt( vector<int>& w, vector<int>& p, int W, int P, int num ){
	// 将num个物品转化为 O(num) 数量级的物品数
	int k(0);
	while( (1<<(k+1)) <= num ){ //请确保W 大于1哦
		w.push_back( (1<<k)*W );
		p.push_back( (1<<k)*P );
		k++;
	}
	w.push_back( (num-(1<<k)+1)*W );	
	p.push_back( (num-(1<<k)+1)*P );
}