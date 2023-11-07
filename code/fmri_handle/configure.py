from torch.utils.data import Dataset

root_dir = ""
kamitani_Aug = f'{root_dir}/Kamitani/augment'
kamitani_Sti = f'{root_dir}/Kamitani/stimulus'
GenericObjectDecoding_dataset_dir = f"{root_dir}/GenericObjectDecoding/"
kamitani_sti_trainID = f"{root_dir}/imageID_training.csv"
kamitani_sti_testID = f"{root_dir}/imageID_test.csv"
GenericObjectDecoding_subs = {
    'sub-1':'Subject1.h5',
    'sub-2':'Subject2.h5',
    'sub-3':'Subject3.h5',
    'sub-4':'Subject4.h5',
    'sub-5':'Subject5.h5'
}
roi_list = {
    'VC':  'ROI_VC',
    'LVC': 'ROI_LVC',
    'HVC': 'ROI_HVC',
    'V1':  'ROI_V1',
    'V2':  'ROI_V2',
    'V3':  'ROI_V3',
    'V4':  'ROI_V4',
    'LOC': 'ROI_LOC',
    'FFA': 'ROI_FFA',
    'PPA': 'ROI_PPA',
}
GOD_fMRI_dim = {'sub-1': 4466, 
                'sub-2': 4404, 
                'sub-3': 4643, 
                'sub-4': 4133, 
                'sub-5': 4370
                }

class fMRI_natural_Dataset(Dataset):
    def __init__(self, data, labels):    
        self.data = data
        self.labels = labels

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

    def __len__(self):
        return len(self.data)




