def dfs(root,prod,graph,visited,val):
	global ans
	children = graph[root]
	if len(children)==1:
		if visited[children[0]]==1:
			ans = max(ans,prod*val[root])
			return

	for i in range(len(children)):
		if visited[children[i]]!=1:
			visited[children[i]]=1
			dfs(children[i],prod*val[root],visited,val)

	return

N = int(input())
root = int(input())
graph = [[] for i in range(N)]
val = list(map(int,input().split()))
for i in range(N-1):
	a,b = list(map(int,input().split()))
	a -= 1
	b -=1
	graph[a].append(b)
	graph[b].append(a)
visited = [0]*N
visited[root]=1
global ans
ans = -float('inf')
dfs(root,1,graph,visited,val)
