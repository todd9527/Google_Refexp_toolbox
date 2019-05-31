from torch.utils.data.dataset import Dataset
from torchvision import transforms
from google_refexp_py_lib.refexp import Refexp
from torch.utils.data import DataLoader


class CaptionDataset(Dataset):
    def __init__(self):
        refexp_filename = 'google_refexp_dataset_release/google_refexp_train_201511_coco_aligned.json'
        coco_filename = 'external/coco/annotations/instances_train2014.json'
        self.refexp = Refexp(refexp_filename, coco_filename)
        super().__init__()

    def __len__(self):
        return len(self.refexp.refexpIds)

    def __getitem__(self, index):
        refexp_id = self.refexp.refexpIds[index]
        annotation_id = self.refexp.refexpToAnnId[refexp_id]
        image_id = self.refexp.annToImgId[annotation_id]

        # todo: add transforms
        return self.refexp.loadImgs([image_id]), self.refexp.loadRefexps([refexp_id]), self.refexp.loadAnns([annotation_id])


# Testing code
dataset = CaptionDataset()
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
for img, refexp, ann in dataloader:
    print(img.shape)
    break
