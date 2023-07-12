import torch


class Config:
    def __init__(self):
        # 数据相关
        self.PAD_TOKEN = '[PAD]'
        self.PAD_ID = 4
        self.BOS_ID = 1
        self.EOS_ID = 2
        self.UNK_ID = 3
        self.NAME_TOKEN = 'xname'
        self.NEAR_TOKEN = 'xnear'
        self.max_src_len = 20
        self.max_target_len = 80
        self.train_dataset_path = 'e2e_dataset/trainset.csv'
        self.dev_dataset_path = 'e2e_dataset/devset.csv'
        self.test_dataset_path = 'e2e_dataset/testset.csv'

        # 训练相关
        self.lr = 0.1
        self.batch_size = 256
        self.n_epoch = 30
        self.val_num = 3  # 每val_num轮进行一次验证
        self.device = torch.device('cuda')

        # 模型相关
        # 词嵌入层
        self.embedding_dim = 256
        self.embedding_dropout = 0.1
        # 编码器
        self.encoder_input_dim = 256
        self.encoder_output_dim = 512
        # 解码器
        self.decoder_input_dim = 512
        self.decoder_output_dim = 512



