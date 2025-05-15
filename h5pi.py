import h5py

class Patch_Exp_Dataset():
    """
    HDF5 포맷 이미지 유전자 데이터를 읽고, 랜덤 샘플링과 전처리를 진행 
    h5_img_path = HDF5 이미지 저장 경로
    h5_df_path = HDF5 유전자 및 scale 저장 경로 
    transforms = 이미지 전처리 함수 
    """
    def __init__(self, h5_img_path, h5_df_path, transforms=None):
        self.h5_img_path = h5_img_path
        self.h5_df_path  = h5_df_path
        self.img_transform = transforms

        with h5py.File(h5_df_path, "r") as f_df:
            self.samples = list(f_df.keys())

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # HDF5 파일 핸들
        if not hasattr(self, 'h5_img'):
            self.h5_img = h5py.File(self.h5_img_path, 'r', swmr=True) # swmr = True : Single Writer Multiple Reader mode (다른 프로세스에서 읽기만 가능)
        if not hasattr(self, 'h5_df'):
            self.h5_df  = h5py.File(self.h5_df_path,  'r', swmr=True)

        patches = len(self.h5_df[sample]['gene'])

        # Random 64 spots selection
        if patches > 128: 
            orig_idx = np.random.choice(patches, 128, replace=False)
        else:
            orig_idx = np.arange(patches)

        # HDF5는 오름차순 인덱스만 허용하므로 정렬 후에 읽기
        sorter     = np.argsort(orig_idx)
        sorted_idx = orig_idx[sorter]

        genes_sorted  = self.h5_df[sample]['gene'][sorted_idx]
        scales_sorted = self.h5_df[sample]['scale'][sorted_idx]
        imgs_sorted   = self.h5_img[sample][sorted_idx]

        inv_sorter = np.argsort(sorter)
        genes  = genes_sorted[inv_sorter]
        scales = scales_sorted[inv_sorter]
        imgs   = imgs_sorted[inv_sorter]

        # 이미지가 uint8 numpy array 형식으로 저장되어 있어 PIL.Image로 변환하고 transform 적용
        if self.img_transform:
            imgs = torch.stack([self.img_transform(Image.fromarray(im)) for im in imgs ])
        else:
            imgs = torch.tensor(imgs)

        genes  = torch.tensor(genes)
        scales = torch.tensor(scales)

        return genes, scales, imgs

if __name__ == "__main__" :
    print(h5py.__version__)