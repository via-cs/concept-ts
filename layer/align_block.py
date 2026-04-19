import torch
import torch.nn as nn
import os

class align(nn.Module):
    def __init__(self):
        super(align, self).__init__()

    def forward(self, x, embedding, output_length):
        idx = x[:, x.size(1) - output_length, :]
        start_idx = idx[:, 6].long() * 4 + (idx[:, 7].long() // 15)

        roll_offsets = (-start_idx) % output_length

        for i in range(x.size(0)):
            embedding[i] = torch.roll(embedding[i], shifts=int(roll_offsets[i].item()), dims=0)
        
        return embedding
    
    def backward(self, x, ans, output_length):
        idx = x[:, x.size(1) - output_length, :]
        start_idx = idx[:, 6].long() * 4 + (idx[:, 7].long() // 15)

        roll_offsets = start_idx % output_length

        for i in range(len(ans)):
            for j in range(x.size(0)):
                ans[i][j] = torch.roll(ans[i][j], shifts=int(roll_offsets[j].item()), dims=0)

        return ans


# class align(nn.Module):
#     def __init__(self):
#         super(align,self).__init__()
#         pass

#     def forward(self,x,embeding,output_length):
#         batch_size=embeding.size()[0]
#         length=embeding.size()[1]
#         start_index=[]
#         for i in range(batch_size):
#             linshi1=x[i][x.size()[1]-output_length][6].item()
#             linshi2=x[i][x.size()[1]-output_length][7].item()
#             linshi=int(linshi1)*4+int(linshi2)//15
#             start_index.append(linshi)

#         all_embeding=[]

#         for i in range(batch_size):
#             now_embeding=torch.index_select(embeding,0,torch.tensor([i]).to(os.environ.get('device')))
#             if start_index[i]==0:
#                 all_embeding.append(now_embeding)
#             else:
#                 first_embeding=torch.index_select(now_embeding,1,torch.tensor([j for j in range(output_length-start_index[i],output_length)]).to(os.environ.get('device'))) 
#                 second_embeding=torch.index_select(now_embeding,1,torch.tensor([j for j in range(output_length-start_index[i])]).to(os.environ.get('device')))
#                 new_embeding=torch.concat([first_embeding,second_embeding],dim=1).reshape(1,length,-1)
#                 all_embeding.append(new_embeding)       

#         embeding=torch.concat(all_embeding,dim=0)
#         self.start_index=start_index
#         return embeding

#     def backward(self,ans,output_length):
#         for j in range(len(ans)):
#             batch_size=ans[j].size()[0]
#             length=ans[j].size()[1]
#             all_ans=[]
#             for i in range(batch_size):
#                 now_ans=torch.index_select(ans[j],0,torch.tensor([i]).to(os.environ.get('device')))
#                 if self.start_index[i]==0:
#                     all_ans.append(now_ans)
#                 else:
#                     second_ans=torch.index_select(now_ans,1,torch.tensor([j for j in range(self.start_index[i])]).to(os.environ.get('device')))
#                     first_ans=torch.index_select(now_ans,1,torch.tensor([j for j in range(self.start_index[i],output_length)]).to(os.environ.get('device')))
#                     new_ans=torch.concat([first_ans,second_ans],dim=1)
#                     all_ans.append(new_ans)
#             ans[j]=torch.concat(all_ans,dim=0)

#         return ans

