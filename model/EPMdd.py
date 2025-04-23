class EP_Mdd(nn.Module):
    def __init__(self):
        super().__init__()
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.face_porb = TwoLayerProbNet(768, 128)
        self.vocal_porb = TwoLayerProbNet(768, 128)

        self.face_GRUAtt = GRUTimeResidualAttentionModel(input_size=136, hidden_size=136)
        self.vocal_GRUAtt = GRUTimeResidualAttentionModel(input_size=25, hidden_size=25)

        self.face_global = GlobalTimeCrossAttention(in_dim=136, h1_dim=128, num_heads=4)
        self.vocal_global = GlobalTimeCrossAttention(in_dim=25, h1_dim=128, num_heads=4)

        self.face_vacal_CroAtt = FaceVocalCrossAttention(in_dim=128, h1_dim=128, num_heads=4)

        self.vacal_face_CroAtt = FaceVocalCrossAttention(in_dim=128, h1_dim=128, num_heads=4)

        self.conv1d1_sp = nn.Conv1d(in_channels=300, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv1d2_tim = nn.Conv1d(in_channels=256, out_channels=64, kernel_size=3, stride=1, padding=1)
        
        self.fc_1 = nn.Linear(64*64, 128)
        self.drop_1 = nn.Dropout(0.1)
        self.fc_2 = nn.Linear(128, 64)
        self.fc_3 = nn.Linear(64, 2)
        
        self.relu_1 = nn.LeakyReLU(negative_slope=0.01) 

        self.conv3_B_data_V_D = nn.Conv1d(in_channels=300, out_channels=300, kernel_size=3, stride=1, padding=1)
        self.conv3_B_data_A_D = nn.Conv1d(in_channels=300, out_channels=300, kernel_size=3, stride=1, padding=1)

        self.conv3_GRU_data_face = nn.Conv1d(in_channels=300, out_channels=300, kernel_size=3, stride=1, padding=1)
        self.conv3_GRU_data_vocal = nn.Conv1d(in_channels=300, out_channels=300, kernel_size=3, stride=1, padding=1)

        self.norm2 = norm_layer(128)
        mlp_hidden_dim = int(128 * mlp_ratio)
        self.mlp = Mlp(in_features=128, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=0.1)


    def forward(self, images, EP_data_face, EP_data_vocal, BJ_trainorTest):

        V_D = images[:, :, 0:136]
        A_D = images[:, :, 136:]

        V_D = self.conv3_B_data_V_D(V_D)
        A_D = self.conv3_B_data_A_D(A_D)

        if BJ_trainorTest == 'Train':
            new_data_face = self.face_porb(EP_data_face, V_D)
            new_data_vocal = self.vocal_porb(EP_data_vocal, A_D)
        elif BJ_trainorTest == 'Test':
            new_data_face = V_D
            new_data_vocal = A_D

        GRU_data_face = self.face_GRUAtt(new_data_face)
        GRU_data_vocal = self.vocal_GRUAtt(new_data_vocal)

        GRU_data_face = self.conv3_GRU_data_face(GRU_data_face)
        GRU_data_vocal = self.conv3_GRU_data_vocal(GRU_data_vocal)

        global_data_face = self.face_global(GRU_data_face, GRU_data_face)  # B 300 128
        global_data_vocal = self.vocal_global(GRU_data_vocal, GRU_data_vocal) # B 300 128
        
        data_face_vocal = self.face_vacal_CroAtt(global_data_face, global_data_vocal)
        data_face_vocal = data_face_vocal + self.mlp(self.norm2(data_face_vocal))
        data_vocal_face = self.vacal_face_CroAtt(global_data_vocal, global_data_face)
        data_vocal_face = data_vocal_face + self.mlp(self.norm2(data_vocal_face))

        x_concat = torch.cat((data_face_vocal, data_vocal_face), dim=2)

        x_ = self.conv1d1_sp(x_concat).transpose(1, 2)
        x_ = self.conv1d2_tim(x_).transpose(1, 2)

        x = x_.reshape(x_.size(0), -1)
        x = self.fc_1(x)
        x = self.drop_1(x)
        x = self.relu_1(x)
        x = self.fc_2(x)
        x = self.drop_1(x)
        x = self.relu_1(x)
        x = self.fc_3(x)

        return x
