import torch 
x = torch.tensor([1.0,2.0, 3.0]) #创建张量，也就是数组
y = x * 2 + 1 
print(y)

#结果：tensor([3., 5., 7.])

#然后搬到GPU上运算
x_gpu = x.to('cuda') #搬到GPU上
y_gpu = x_gpu * 2 + 1
print(y_gpu)

#结果：tensor([3., 5., 7.], device='cuda:0')