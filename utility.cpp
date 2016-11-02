#include<algorithm> // for sort()
#include<vector>



// ÀëÉ¢»¯a:  1 1 1000 1000 -> 2 2 4 4
int a[maxn];
bool cmp( int x, int y ){
	return a[x] < a[y];
}
void discretization(int n){
	vector<int> ID(n, 0);
	for( int i(0); i < n ; i++ ) ID[i] = i;
	sort(ID.begin(), ID.end(), cmp );
	for( int i(0); i < n ; ){
		int j(i);
		while( j < n && a[ID[j]]==a[ID[i]] ) j++;
		while( i<j ) a[ID[i++]] = j-1;
	}
}