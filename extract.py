"""
tag-to-coulmn (.excel)
"""
from PIL import Image
import os
import sys
import pickle
import json
import os
import re

from tqdm import tqdm
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import xlsxwriter


def json_to_pickle(file_name):
    """
    json 파일을 pkl 파일로 변환.
    :param file_name: 파일 이름
    """
    with open('./data/objects.json', 'r') as fr:
        json_data = json.load(fr)

    with open(file_name, 'wb') as fw:
        pickle.dump(json_data, fw)

    sys.stdout.write('pickling done ... \n')


def name_tokenize(object_name, lm):
    """
    object name을 토크나이즈 해주는 함수
    :param object_name: str
    :param lm: WordNetLemmatizer
    :return: object_name: str
    """
    # print('before object_name', object_name)
    object_name = re.sub("[^a-zA-Z0-9\s_']", "", object_name).strip()
    object_name = re.sub("[0-9]+", "num" + object_name, object_name)
    # print('after object name', object_name)
    try:
        if object_name[0] == ' ':
            object_name = object_name[1:]
        if '-' in object_name:
            object_name = object_name.replace('-', '_')
        if ' ' in object_name:
            object_name = object_name.replace(' ', '_')
        if len(object_name) <= 1:
            return None
    except IndexError:
        print("IndexError", object_name)
        return None
    # object_name = stemmer.stem(object_name)
    object_name = lm.lemmatize(object_name, pos='n')
    return object_name


def extract_object_tag(visual_genome_data):
    """
    json or pkl 파일일 읽어 object와 bbox(region)를 추출
    엑셀에 저장할 worksheet에 저장하는 함수
    :param visual_genome_data:
    :return: object와 object에 담긴 region 정보를 가진 Dict[List]
    """
    # stemmer = LancasterStemmer()
    lm = WordNetLemmatizer()

    # visual_genome_data = json.load(json_data)

    workbook = xlsxwriter.Workbook('./data/test.csv')
    worksheet = workbook.add_worksheet()
    worksheet.write(0, 0, 'image_id')

    object_region = {}
    object_max_num = 0
    for idx, data in enumerate(tqdm(visual_genome_data[:300])):
        object_id_name = {}
        image_id = data['image_id']
        regions = []
        for objects in data['objects']:
            object_name = name_tokenize(objects['names'][0], lm)
            if object_name is None:
                continue
            region = {
                'name': object_name,
                'x': objects['x'],
                'y': objects['y'],
                'w': objects['w'],
                'h': objects['h']
            }
            regions.append(region)
            object_id_name[objects['object_id']] = object_name
        tag_list = sorted(set(object_id_name.values()))
        object_region[image_id] = regions
        if len(tag_list) > object_max_num:
            object_max_num = len(tag_list)

        worksheet.write(idx + 1, 0, image_id)
        for set_idx, tag in enumerate(tag_list):
            worksheet.write(idx + 1, set_idx + 1, tag)

    for num in range(object_max_num):
        worksheet.write(0, num + 1, 'COL' + str(num))
    workbook.close()
    return object_region


def visualize_image(image_tag):
    image_id = list(image_tag.keys())[0]
    img = Image.open('./data/VG_100K/'+str(image_id)+'.jpg')
    regions = image_tag[image_id]
    print(regions)
    plt.imshow(img)
    ax = plt.gca()
    for region in regions:
        print(region)
        ax.add_patch(Rectangle((region['x'], region['y']),
                               region['w'],
                               region['h'],
                               fill=False,
                               edgecolor='red',
                               linewidth=3))
        ax.text(region['x'], region['y'], region['name'], style='italic',
                bbox={'facecolor': 'white', 'alpha': 0.7, 'pad': 10})
    fig = plt.gcf()
    plt.tick_params(labelbottom='off', labelleft='off')
    plt.show()


def init(file_name):
    if not os.path.exists(file_name):
        json_to_pickle(file_name)


def main():
    file_name = './data/objects.pkl'
    init(file_name)
    with open(file_name, 'rb') as fr:
        visual_genome_data = pickle.load(fr)
    with open('./data/objects.json', 'r') as fr:
        visual_genome_data = json.load(fr)
    image_tag = extract_object_tag(visual_genome_data)
    visualize_image(image_tag)


main()
