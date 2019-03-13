from process.data_helper import *

def save_image_reprezentation():
    label_dict,_ = load_CLASS_NAME()

    save_dir = r'./IDs'
    LIST_DIR = r'/data2/shentao/Projects/Kaggle_Whale/image_list'
    fold_index = 0
    train_list = read_txt(LIST_DIR + '/train_fold' + str(fold_index) + '.txt')
    val_list = read_txt(LIST_DIR + '/val_fold' + str(fold_index) + '.txt')

    all_list = train_list + val_list

    bbox_dict = load_bbox_dict_replace_with_modified()

    for name,id in all_list:
        TRN_IMGS_DIR = '/data2/shentao/DATA/Kaggle/Whale/raw/train/'
        TST_IMGS_DIR = '/data2/shentao/DATA/Kaggle/Whale/raw/test/'

        name_ = os.path.join(TRN_IMGS_DIR, name)
        if not os.path.exists(name_):
            name_ = os.path.join(TST_IMGS_DIR, name)
        if not os.path.exists(name_):
            print('Wrong')

        img = get_cropped_img(name_, bbox_dict)

        if id == -1:
            continue

        save_name_tmp = os.path.join(save_dir, label_dict[id]+'.jpg')

        if os.path.exists(save_name_tmp):
            continue

        img = cv2.resize(img,(512,256))
        cv2.imwrite(save_name_tmp, img)

    return

def get_test_sure_label():
    list = os.listdir(r'/data2/shentao/DATA/Kaggle/Whale/raw/onshot_show_913_65')
    list += os.listdir(r'/data2/shentao/DATA/Kaggle/Whale/raw/res_show_913_113')

    list = [tmp.split('_')[3]+'.jpg' for tmp in list]

    leak = './leaks/leaks_0115.csv'
    print(leak)
    parsed = pd.read_csv(leak)
    test = parsed['b_test_img']

    for id in test:
        list.append(id)

    print(len(set(list)))
    return set(list)

def save_csv_to_imgs(csv):
    sure_label = get_test_sure_label()

    print(sure_label)

    test_dir = r'/data2/shentao/DATA/Kaggle/Whale/raw/test_crop/test_crop'
    save_dir = r'./921_not_sure_label_NoNW'

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    id_dir = r'./IDs'
    parsed = pd.read_csv(csv)
    test = parsed['image']
    target = parsed['id']

    c = 0

    for id,label in zip(test,target):
        print(id)
        # print()
        nw = label.split(' ')[0]

        if (not id in sure_label) and nw != 'new_whale':
            imgs = []
            test_path = os.path.join(test_dir, id)
            img = cv2.imread(test_path,1)
            img = cv2.resize(img,(384,192))
            imgs.append(img)

            for tmp in label.split(' '):
                tmp_path = os.path.join(id_dir, tmp+'.jpg')

                if not os.path.exists(tmp_path):
                    img = np.zeros([192,384,3]).astype(np.uint8)
                else:
                    img = cv2.imread(tmp_path, 1)
                    img = cv2.resize(img, (384, 192))

                imgs.append(img)

            img_final = np.concatenate(imgs,axis=0)
            print(img_final.shape)

            save_name = id.replace('.jpg',' ')+label+'.jpg'
            save_name = save_name.replace(' ','_')
            save_name = os.path.join(save_dir,save_name)

            cv2.imwrite(save_name, img_final)
            print(save_name)

        else:
            c += 1
            print(id)

    print(c)

def change_val1():
    txt = r'/data2/shentao/Projects/Kaggle_Whale/image_list/val_fold0.txt'
    list = read_txt(txt)

    dict = {}
    for img, id in list:
        if id in dict:
            dict[id].append(img)
        else:
            dict[id] = [img]

    print(len(dict))


    num_newwhale = 230
    new_whale = dict[-1]
    random.shuffle(new_whale)
    new_whale_val = new_whale[0:num_newwhale]
    new_whale_train = new_whale[num_newwhale:]

    val = []
    for tmp in new_whale_val:
        val.append([tmp, -1])

    ids = dict.keys()
    random.shuffle(ids)

    ids_val = ids[0:512]
    print(ids_val)

    for item in ids_val:
        if item != -1:
            list_tmp = dict[item]
            img = list_tmp[0]
            id = item
            val.append([img,id])

    print(len(val))

    new_val = r'/data2/shentao/Projects/Kaggle_Whale/image_list/val_qj.txt'
    f = open(new_val, 'w')

    for img, id in val:
        f.write(img+' '+str(id)+'\n')
    f.close()

    print('done')

def save_crop_imgs():
    TRN_IMGS_DIR = '/data2/shentao/DATA/Kaggle/Whale/raw/train/'
    TST_IMGS_DIR = '/data2/shentao/DATA/Kaggle/Whale/raw/test/'

    all_dir = r'/data2/shentao/DATA/Kaggle/Whale/raw/train_all_crop_/'

    LIST_DIR = r'/data2/shentao/Projects/Kaggle_Whale/image_list'
    train_list = read_txt(LIST_DIR + '/train_fold0.txt')
    val_list = read_txt(LIST_DIR + '/val_fold0.txt')
    all_list = train_list + val_list

    bbox_dict = load_bbox_dict_replace_with_modified()

    def save_list(list, dir):

        for name, id in list:
            name_ = os.path.join(TRN_IMGS_DIR, name)
            if not os.path.exists(name_):
                name_ = os.path.join(TST_IMGS_DIR, name)
            if not os.path.exists(name_):
                print('Wrong')

            img = get_cropped_img(name_,bbox_dict)

            save_dir_tmp = os.path.join(dir)
            if not os.path.exists(save_dir_tmp):
                os.makedirs(save_dir_tmp)

            save_name_tmp = os.path.join(save_dir_tmp,name)
            cv2.imwrite(save_name_tmp, img)

    save_list(all_list, all_dir)

def get_aligned_pts(csv):
    pts_data = pd.read_csv(csv)
    labelName = pts_data['key_id'].tolist()
    pts = pts_data['word'].tolist()

    pts_final = []
    for tmp in pts:
        tmp = tmp.split(' ')
        pt_list = []
        for pt in tmp:
            if '_' in pt:
                x,y = pt.split('_')
                pt_list.append([int(x), int(y)])

        pts_final.append(pt_list)

    return labelName,pts_final

def load_alignpoint_dict():
    pts_dict = {}

    csv = r'/data1/shentao/Projects/deep_reid/whale/process/r101_flip_train.csv'
    labelName, pts = get_aligned_pts(csv)
    for (name, pt) in zip(labelName, pts):
        pts_dict[name] = pt

    csv = r'/data1/shentao/Projects/deep_reid/whale/process/r101_flip_test.csv'
    labelName, pts = get_aligned_pts(csv)
    for (name, pt) in zip(labelName, pts):
        pts_dict[name] = pt

    return pts_dict

def get_bbox_from_xml(file):
    tree = etree.parse(file)
    bbox_list = []
    for bbox in tree.xpath('//bndbox'):
        for corner in bbox.getchildren():
            bbox_list.append(int(corner.text))
        break

    return bbox_list

def get_modified_bbox(dir = r'./modified_bbox_1224/'):
    dict = {}
    list = os.listdir(dir)

    for csv in list:
        bbox = get_bbox_from_xml(os.path.join(dir, csv))
        dict[csv.replace(r'.xml',r'.jpg')] = bbox

    return dict

def new_split():
    NoNW_num = 400
    NW_num = 200

    tmp = load_train_map()
    print(len(tmp))

    id_dict = {}
    for item in tmp:
        id = tmp[item][0]
        if id not in id_dict:
            id_dict[id] = [item]
        else:
            id_dict[id].append(item)

    non_oneshot_list = []
    for id in id_dict:
        if len(id_dict[id]) > 1:
            non_oneshot_list.append(id)
    print(len(non_oneshot_list))

    random.shuffle(non_oneshot_list)
    val_path = PJ_DIR + r'/image_list/val4.txt'

    f = open(val_path, 'w')
    for i in range(NoNW_num):
        id = non_oneshot_list[i]
        random.shuffle(id_dict[id])
        img = id_dict[id][0]
        f.write(img + ' ' + str(id) + '\n')

    for i in range(NW_num):
        id = -1
        random.shuffle(id_dict[id])
        img = id_dict[id][0]
        f.write(img + ' ' + str(id) + '\n')
    f.close()