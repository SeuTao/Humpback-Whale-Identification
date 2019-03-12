from collections import defaultdict, Counter
from process.data_helper import *
SIGFIGS = 6

id_name_label, dict_label = load_CLASS_NAME()
dict_label['new_whale'] = -1
id_name_label[-1] = 'new_whale'

def read_models(model_weights, thres, blend=None):
    if not blend:
        blend = defaultdict(Counter)

    count = 0
    for m, w in model_weights.items():
        m_list = os.listdir(m)

        print(m)
        print(len(m_list))

        for m_tmp in m_list:
            count += 1
            with open(os.path.join(m, m_tmp), 'r') as f:
                f.readline()

                for l in f:
                    id, r = l.split(',')
                    id, r = id, r.split(' ')

                    n = len(r)//2 * 2
                    for i in range(0, n, 2):
                        tmp = r[i]
                        prob = float(r[i + 1])
                        k = dict_label[tmp]
                        v = 10**(SIGFIGS - 1) * prob
                        blend[id][k] += w * v

                    # add new_whale
                    k = dict_label['new_whale']
                    v = 10 ** (SIGFIGS - 1) * float(thres)
                    blend[id][k] += w * v

    print(count)
    return blend

def clalibrate_distribution(blend):
    id_dict = {}
    id_top1_dict = {}
    for i in range(5005):
        id_top1_dict[i] = [None, 0.0]

    count_NewWhale = 0

    for id, v in blend.items():
        for t in enumerate(v.most_common(1)):
            if t[1][0] == -1:
                count_NewWhale += 1

            if t[1][0] not in id_dict:
                id_dict[t[1][0]] = [id]
            else:
                id_dict[t[1][0]].append(id)

        for t in enumerate(v.most_common(5)):
            if t[1][0] != -1:
                if t[1][1] > id_top1_dict[t[1][0]][1]:
                    id_top1_dict[t[1][0]] = [id, t[1][1]]

    print('id num:' + str(len(id_dict)))
    print('missing id num:' + str(5005 - len(id_dict)))
    print('new whale num:'  + str(count_NewWhale))

    missing_ids = {}
    for i in range(5005):
        if i not in id_dict:
            missing_ids[i] = 0
    return blend, missing_ids

def write_models(blend, file_name, is_top1 = False):
    with open( file_name + '.csv', 'w') as f:
        f.write('image,id\n')
        nc = 0
        for id, v in blend.items():
            if is_top1:
                l = ' '.join(['{}'.format(id_name_label[int(t[0])]) for i_t, t in enumerate(v.most_common(20)) if i_t < 1])
                l += ' None None None None'
                print(l)
                f.write(','.join([str(id), l + '\n']))
            else:
                l = ' '.join(['{}'.format(id_name_label[int(t[0])]) for i_t, t in enumerate(v.most_common(20)) if i_t < 5])
                f.write(','.join([str(id), l + '\n']))

            if l.find('new_whale')==0:
                nc += 1
        print('new whale num: '+ str(nc))
    return file_name + '.csv'

if __name__ == '__main__':
    model_pred = {
        r'/data1/shentao/competitions/whale/models/seresnet101_final_loss_256X512_pseudo/checkpoint/max_valid_model': 10,
    }
    thres =  0.18
    avg = read_models(model_pred, thres)
    avg, missing_ids = clalibrate_distribution(blend=avg)
    csv_name = write_models(avg, 'thres_0.18_4066_2140', is_top1=False)