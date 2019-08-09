本文件夹下的数据对应spatial2

1. training set
   每个frame帧，包含20个视点，这20个视点环绕一圈
   共: 40*200 = 8000组训练样本
   
   没有coherence
   每次从这个20组视点中选n(2-5)个视点作为observation，这个n个视点均匀分布，环绕一圈
   再从这20个视点中任选1个作为query
   
   n_views = random.randint(2,5)
   indices = torch.randperm(m)
   indices = indices.sort()[0]
   view_index = self.compute_view_index(n_views,m)
   
   def compute_view_index(self,n_views,m):
       view_bias = random.randint(0,m-1)
       step = m//n_views
       start = 0
       view_index = [1] * n_views
       for g in range(0,n_views):
           index = start + step*g
           view_index[g] = index
       for g in range(0,n_views):
           view_index[g] = (view_index[g] + view_bias)%m
       view_index = sorted(view_index)
       return view_index
	   
    representation network是tower
	
2. 实验条件都同1
   representation network是pool
 