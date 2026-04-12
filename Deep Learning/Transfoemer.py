import torch
import torch.nn as nn
import math
import numpy as np


class MultHeadAttention(nn.Module):
    """
        多头注意力机制
        d_model  通常为512
        num_head  通常为8
    """

    def __init__(self, d_model, num_heads):
        """
        初始化 :    初始化头的数量 以及全连接层数
                   初始化 QKV 三种权重矩阵进行全连接化  Output
        :param d_model:
        :param num_heads:
        """
        super(MultHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_head = num_heads
        self.d_k = d_model // num_heads

        self.W_Q = nn.Linear(d_model, d_model, bias=True)
        self.W_K = nn.Linear(d_model, d_model, bias=True)
        self.W_V = nn.Linear(d_model, d_model, bias=True)
        self.W_O = nn.Linear(d_model, d_model, bias=True)

    """
       计算输出值
       在计算输出值时 利用掩码  masked_fill(mask==0,-1e9)
       后用softmax获取概率最大值的值转化为概率获取相似度
       用相似度与value 矩阵 与相应的值进行对应返回       

    """

    def computer_output(self, Q, V, K, mask=None):
        score = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            score = score.masked_fill(mask == 0, -1e9)  # 填充掩码 布尔掩码  将0位置的元素填充为-1e9
        score = torch.softmax(score, dim=-1)
        output = torch.matmul(score, V)
        return output

    """
         改变输入特征值的结构shape   
    """

    def spilt_head(self, x):
        batch_size, seq_len, d_model = x.size()
        return x.view(batch_size, seq_len, self.num_head, self.d_k).transpose(1, 2)

    def Combine_head(self, x):
        batch_size, seq_len, d_model = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)  # contiguous为浅拷贝 与源数据公用一个内存

        # 输出后的x.shape(batch_size,seq_len,d_model)

    def forward(self, Q, K, V, mask=None):
        Q = self.spilt_head(self.W_Q(Q))
        K = self.spilt_head(self.W_K(K))  # 线性映射 多头注意力并行计算
        V = self.spilt_head(self.W_V(V))

        output = self.computer_output(Q, V, K, mask)  # 计算 输出值
        output = self.W_O(self.Combine_head(output))  # 合头进行总输出
        return output


class positionalFeedForward(nn.Module):
    """

         位置前馈网络    旨在输入位置信息，使得每个位置都有独特性，增加泛化能力，便于后续直接将位置信息添加进每个特征值的特征向量中
         使得模型 学习到词中信息

    """

    def __init__(self, d_model, d_ff):
        super(positionalFeedForward, self).__init__()
        self.W_1 = nn.Linear(d_model, d_ff)
        self.W_2 = nn.Linear(d_ff, d_model)
        self.activation = nn.ReLU()

    def forward(self, x):
        return self.W_2(self.activation(self.W_1(x)))


class PositionalEncoding(nn.Module):
    """
            位置编码  对每个词的位置信息进行编码
            pe[:, 0::2] = np.sin(position * div_term) # 偶数维度使用sin
            pe[:, 1::2] = np.cos(position * div_term) # 奇数维度使用cos
    """

    def __init__(self, d_model, max_len):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        position = torch.arange(0, max_len, dtype=torch.float64)
        Formula = np.exp(torch.arange(0, d_model) * (-math.log(10000) / d_model))
        self.pe = torch.zeros(max_len, d_model)
        self.pe[:, 0::2] = np.sin(position * Formula)
        self.pe[:, 1::2] = np.cos(position * Formula)

    def Forward(self, x):
        return x + self.pe[:x.size(1), :]

    """
     # 将位置信息放入特征向量中 self.pe[:x.size(1),:] 将当前已知的位置编码加入,防止造成max_len与seq_len 向量维度不同而造成的报错

    """


class Encoder(nn.Module):
    """
        编码器层  需要利用  多头注意力机制 以及位置前馈网络 将前面生成的位置序列正向传播下去 并且利用归一化层
        残差连接将后续序列进行处理

    """

    def __init__(self, d_model, dropout, d_ff, num_head=8):
        super(Encoder, self).__init__()
        self.attention = MultHeadAttention(d_model, num_head)
        self.FeedForward = positionalFeedForward(d_model, d_ff)
        self.Norm = nn.LayerNorm(d_model)
        self.Norm1 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask):
        """
         层归一化和残差连接 如果需要添加子层 需要在额外层中 添加层归一化以及残差连接
        :param x:
        :param mask:
        :return:
        """
        attention = self.attention.forward(x, x, x, mask)
        x = self.Norm(x + self.drop(attention))

        out = self.FeedForward(x)
        x = self.Norm1(x + self.drop(out))

        return x


class Decoder(nn.Module):
    """
        与编码器原理相似  多了一个 交叉注意力机制  利用解码器掩码进行预测  注意多出来的每一层都需要进行残差连接以及层归一化


    """

    def __init__(self, d_model, dropout, d_ff, num_head):
        super(Decoder, self).__init__()
        self.attention = MultHeadAttention(d_model, num_head)  # 多头注意力机制
        self.Cross_attention = MultHeadAttention(d_model, num_head)  # 交叉注意力机制
        self.FeedForward = positionalFeedForward(d_model, d_ff)
        self.Norm = nn.LayerNorm(d_model)
        self.Norm1 = nn.LayerNorm(d_model)
        self.Norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    """
            decoder_state: (batch, tgt_len, d_model) - 解码器当前状态
            encoder_output: (batch, src_len, d_model) - 编码器输出
    """

    def forward(self, x, decoder_state, encoder_output, mask):
        attention = self.attention(x, x, x, mask)
        x = self.Norm(x + self.drop(attention))

        cross_attention = self.Cross_attention(decoder_state, encoder_output, encoder_output, mask)
        x = self.Norm(x + self.drop(cross_attention))

        Feed_out = self.FeedForward(x)
        x = self.Norm2(x + self.drop(Feed_out))

        return x


class Transformer(nn.Module):
    """
        实现Transformer
        首先 需要生成填充编码 以及 未来编码
        以 mask作为指标来对value进行训练

    """

    def __init__(self, d_model, num_head, dropout, d_ff, out_size, max_len, Input_size, Out_size, num_layers, Dropout):
        super(Transformer).__init__()
        self.encoder = nn.Embedding(Input_size, d_model)
        self.decoder = nn.Embedding(Out_size, d_model)
        self.encoder_layers = nn.ModuleList(
            [Encoder(d_model, num_head, d_ff, dropout) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList(
            [Decoder(d_model, num_head, d_ff, dropout) for _ in range(num_layers)])
        self.position = PositionalEncoding(d_model, max_len)
        self.output = nn.Linear(d_model, out_size)
        self.dropout = nn.Dropout(Dropout)

    def generate(self, mask_encoder, mask_decoder):
        encoder_mask = (mask_encoder != 0).unsqueeze(1).unsqueeze(2)
        decoder_mask = (mask_decoder != 0).unsqueeze(1).unsqueeze(3)

        seq_length = mask_decoder.size(1)
        future_mask = (torch.tril(torch.ones(1, 1, seq_length, seq_length), diagonal=-1)).bool()  # tril 不使用默认值则不保留对角线
        # 负数是指往左下方移动对角线既下三角
        decoder_mask = future_mask & decoder_mask

        return encoder_mask, decoder_mask

    def forward(self, encoder, decoder):
        encoder_mask, decoder_mask = self.generate(encoder, decoder)

        output = self.dropout(self.position(self.encoder(encoder)))
        sample = output
        for i in self.encoder_layers:
            sample = i(sample, encoder_mask)

        de_output = self.dropout(self.position(self.decoder(decoder)))
        de_sample = de_output
        for j in self.decoder_layers:
            de_sample = j(de_output, output, encoder_mask, decoder_mask)

        Result = self.output(de_sample)

        return Result


if __name__ == "__main__":
    pass






