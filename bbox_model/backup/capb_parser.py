# import numpy as np
# from sklearn.model_selection import train_test_split
# import pandas as pd
#
# class CAPB():
#     def __init__(self, data_dir, set_type, train_val):
#         # set_name should be in ['blouse' 'dress' 'outwear' 'skirt' 'trousers']
#         self.data_dir = data_dir
#         set_df = pd.read_table(data_dir + 'Eval/list_eval_partition.txt', delim_whitespace=True, skiprows=1)
#         set_df = set_df[set_df['evaluation_status']==train_val]
#         cate_cloth_df = pd.read_table(data_dir + 'Anno/list_category_cloth.txt', delim_whitespace=True, skiprows=1)
#         cate_idcs = (cate_cloth_df.index[cate_cloth_df['category_type'] == set_type]+1).tolist()
#         cate_img_df = pd.read_table(data_dir + 'Anno/list_category_img.txt', delim_whitespace=True, skiprows=1)
#         cate_img_df = cate_img_df[cate_img_df['category_label'].isin(cate_idcs)]
#         bbox_df = pd.read_table(data_dir + 'Anno/list_bbox.txt', delim_whitespace=True, skiprows=1)
#         bbox_df = bbox_df[bbox_df['image_name'].isin(set_df['image_name'])]
#         self.anno_df = bbox_df[bbox_df['image_name'].isin(cate_img_df['image_name'])]
#
#     def size(self):
#         return len(self.anno_df)
#
#     def get_image_path(self, image_index):
#         row = self.anno_df.iloc[image_index]
#         image_path = self.data_dir + row['image_name']
#         return image_path
#
#     def get_bbox(self, image_index):
#         row = self.anno_df.iloc[image_index]
#         bbox = np.array([[row['x_1'], row['y_1'], row['x_2'], row['y_2']]], dtype=np.float32)
#         return bbox
#
# if __name__ == '__main__':
#     db_path = '/home/storage/lsy/fashion/Category_and_Attribution_Prediction_Benchmark/'
#     for k in [1,2,3]:
#         for t in ['train', 'val', 'test']:
#             kpda = CAPB(db_path, k, t)
#             print('%d '%k + t + ' : %d' % kpda.size())




