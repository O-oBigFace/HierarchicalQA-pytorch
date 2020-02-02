from torch.utils.data import Dataset, DataLoader
import h5py


class MYDataset(Dataset):
    def __init__(self, path_img_h5, path_prepro_h5, feat_type="VGG"):
        img_h5 = h5py.File(path_img_h5, 'r')
        self.img_feat = img_h5.get('images')

        ques_h5 = h5py.File(path_prepro_h5, 'r')
        self.ques = ques_h5.get('ques')
        self.ques_id = ques_h5.get('ques_id')
        self.img_pos = ques_h5.get('img_pos')
        self.ques_len = ques_h5.get('ques_len')
        self.ans = ques_h5.get('ans')

        self.img_feat_dim = 512 if feat_type == 'VGG' else 2048

    def __len__(self):
        return len(self.ques)

    def __getitem__(self, idx):
        sample = dict()
        sample['image'] = self.img_feat[self.img_pos[idx]].reshape((-1, self.img_feat_dim))
        sample['question'] = self.ques[idx]
        sample['ques_id'] = self.ques_len[idx]
        sample['answer'] = self.ans[idx]

        return sample


if __name__ == '__main__':
    myd = MYDataset("../data/vqa_data_img_vgg_val.h5", "../data/cocoqa_prepro_val.h5")
    dataloader = DataLoader(myd, batch_size=4, shuffle=True)

    for i_batch, sample_batched in enumerate(dataloader):
        print(i_batch, sample_batched['image'].size(), sample_batched['question'].size())
        print(sample_batched)
        break
