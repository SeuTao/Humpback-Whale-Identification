from data_helper import *

TRAIN_DF  = []
TEST_DF   = []

class WhaleDataset(Dataset):
    def __init__(self, mode, fold_index='<NIL>', image_size=(128,256),
                 augment = None,
                 is_pseudo = False,
                 is_flip = False):

        super(WhaleDataset, self).__init__()
        self.mode       = mode
        self.augment = augment
        self.is_pseudo = is_pseudo
        self.is_flip = is_flip

        self.pseudo_list = []
        if self.is_pseudo:
            self.pseudo_list = load_pseudo_list()

        self.bbox_dict = load_bbox_dict()
        self.label_dict = load_label_dict(os.path.join(LIST_DIR, r'label_list.txt'))
        self.class_num = len(self.label_dict)
        print(self.class_num)

        self.train_image_path = TRN_IMGS_DIR
        self.test_image_path = TST_IMGS_DIR
        self.image_size = image_size
        self.fold_index = None
        self.set_mode(mode, fold_index)

    def set_mode(self, mode, fold_index):
        self.mode = mode
        self.fold_index = fold_index
        print('fold index set: ', fold_index)

        if self.mode == 'train' or self.mode == 'train_list':
            self.train_list = load_train_list()
            val_list =  read_txt(LIST_DIR + '/new_val'+str(self.fold_index)+'.txt')
            print(LIST_DIR + '/val'+str(self.fold_index)+'.txt')
            val_set = set([tmp for tmp,_ in val_list])
            self.train_list = [tmp for tmp in self.train_list if tmp[0] not in val_set]
            self.train_list += self.pseudo_list
            self.train_dict, self.id_list = image_list2dict(self.train_list)
            self.num_data = len(self.train_list)*2
            print('set dataset mode: train')

        elif self.mode == 'val':
            self.val_list = read_txt(LIST_DIR + '/new_val'+str(self.fold_index)+'.txt')
            print(LIST_DIR + '/val'+str(self.fold_index)+'.txt')
            self.num_data = len(self.val_list)
            print('set dataset mode: val')

        elif self.mode == 'test':
            self.test_list = os.listdir(TST_IMGS_DIR)
            self.test_list = [tmp for tmp in self.test_list if '.jpg' in tmp]
            self.num_data = len(self.test_list)
            print('set dataset mode: test')

        elif self.mode == 'test_train':
            self.test_list = os.listdir(TRN_IMGS_DIR)
            self.test_list = [tmp for tmp in self.test_list if '.jpg' in tmp]
            self.num_data = len(self.test_list)
            print('set dataset mode: test')

        print('data num: ' + str(self.num_data))


    def __getitem__(self, index):
        if self.fold_index is None:
            print('WRONG!!!!!!! fold index is NONE!!!!!!!!!!!!!!!!!')
            return

        if self.mode == 'train':
            if index >= len(self.train_list):
                image_index = index - len(self.train_list)
            else:
                image_index = index

            image_tmp, label = self.train_list[image_index]
            image_path = os.path.join(self.train_image_path, image_tmp)

            if not os.path.exists(image_path):
                image_path = os.path.join(self.test_image_path, image_tmp)

            image = cv2.imread(image_path, 1)
            image = get_cropped_img(image, self.bbox_dict[os.path.split(image_path)[1]], is_mask=False)

        if self.mode == 'train_list':
            if index >= len(self.train_list):
                image_index = index - len(self.train_list)
            else:
                image_index = index
            image_tmp, label = self.train_list[image_index]

            if label == -1:
                return None,label,None

            if index >= len(self.train_list):
                label += 5004

            return None,label,None

        if self.mode == 'val':
            image_tmp,label = self.val_list[index]
            image_path = os.path.join(self.train_image_path, image_tmp)

            if not os.path.exists(image_path):
                image_path = os.path.join(self.test_image_path, image_tmp)

            image = cv2.imread(image_path, 1)
            image = get_cropped_img(image, self.bbox_dict[os.path.split(image_path)[1]], is_mask=False)


        if self.mode == 'test':
            image_path = os.path.join(self.test_image_path, self.test_list[index])
            image = cv2.imread(image_path, 1)
            image = get_cropped_img(image, self.bbox_dict[os.path.split(image_path)[1]], is_mask=False)

            image_id = self.test_list[index]
            image = cv2.resize(image, (self.image_size[1], self.image_size[0]))
            image = aug_image(image, is_infer=True,augment=self.augment)

            image = np.transpose(image, (2, 0, 1))
            image = image.astype(np.float32)
            image = image.reshape([3, self.image_size[0], self.image_size[1]])
            image = image / 255.0

            return image_id, torch.FloatTensor(image)

        if self.mode == 'test_train':
            image = cv2.imread(os.path.join(self.train_image_path, self.test_list[index]), 1)
            image_id = self.test_list[index]
            image = cv2.resize(image, (self.image_size[1], self.image_size[0]))
            image = aug_image(image, is_infer=True,augment=self.augment)

            image = np.transpose(image, (2, 0, 1))
            image = image.astype(np.float32)
            image = image.reshape([3, self.image_size[0], self.image_size[1]])
            image = image / 255.0
            return image_id, torch.FloatTensor(image)

        image = cv2.resize(image, (self.image_size[1], self.image_size[0]))

        if self.mode == 'train':
            if random.randint(0,1) == 0:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

            if random.randint(0, 1) == 0:
                image = Perspective_aug(image)
                image = cv2.resize(image, (self.image_size[1], self.image_size[0]))

            image = aug_image(image)

            if label == -1:
                seq = iaa.Sequential([iaa.Fliplr(0.5)])
                image = seq.augment_image(image)

            elif index >= len(self.train_list):
                seq = iaa.Sequential([iaa.Fliplr(1.0)])
                image = seq.augment_image(image)
                label += 5004

        elif self.mode == 'val':
            image = aug_image(image, is_infer=True,augment=self.augment)

            if self.is_flip:
                seq = iaa.Sequential([iaa.Fliplr(1.0)])
                image = seq.augment_image(image)

                if label != -1:
                    label += 5004

        image = np.transpose(image, (2, 0, 1))
        image = image.astype(np.float32)
        image = image.reshape([-1, self.image_size[0], self.image_size[1]])
        image = image / 255.0

        NW = 0
        if label == -1:
            label = 10008
            NW = 1

        return torch.FloatTensor(image), label, NW

    def __len__(self):
        return self.num_data

# check #################################################################
def run_check_train_data():
    dataset = WhaleDataset(mode = 'train',
                           image_size=(256,768),
                           fold_index=0,
                           is_pseudo=False,
                           augment=[0])

    print(dataset)
    c = 0
    num = len(dataset)
    for m in range(num):
        i = np.random.choice(num)
        image, label, l2 = dataset[i]

        print(image.shape)
        print(label)
        print(label)

        c += 1
        if c == 100:
            break

# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))
    run_check_train_data()












