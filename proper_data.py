import json
import numpy as np

# features = [("word count", 'NUMERIC'), ("average word length", 'NUMERIC'),
#             ("length of the longest word", 'NUMERIC'),
#             ("whether start with number", ['True', 'False']),
#             ("whether start with who/what/why/where/when/how",
#              ['True', 'False']), ("label", ['0', '1'])]
features = []


def extract_features(text):
    doc = text.strip().split(' ')
    f1 = 0
    f2 = 0
    f3 = 0
    f4 = False
    f5 = False
    for token in doc:
        word = token.lower()
        if f1 == 0:
            if word[0].isdigit():
                f4 = True
            if word in ['who', 'what', 'why', 'where', 'when', 'how']:
                f5 = True
        f1 += 1
        length = len(word)
        f2 += length
        f3 = max(f3, length)
    if f1 == 0:
        return False
    return (f1, f2 * 1.0 / f1, f3, f4, f5)


id_features = {}
with open('C:/Users/hayde/OneDrive/Documents/clickbait_project/clickbait-data/instances_train.jsonl', 'r') as f:
    for line in f:
        dic = json.loads(line)
        if len(dic['postText'][0]) > 0:
            feat = extract_features(dic['postText'][0])
            if feat != False:
                id_features.setdefault(dic['id'], feat)
# print(len(id_features))


id_labels = {}
with open('C:/Users/hayde/OneDrive/Documents/clickbait_project/clickbait-data/truth_train.jsonl', 'r') as f:
    for line in f:
        dic = json.loads(line)
        label = 1
        if dic['truthClass'][0] == 'n':
            label = 0
        if dic['id'] in id_features:
            id_labels.setdefault(dic['id'], label)
# print(len(id_labels))

data = {}
data.setdefault('attributes', features)
data.setdefault('description', '')
data.setdefault('relation', 'clickbait_sample')
data.setdefault('data', [])
for i in id_labels:
    tmp = [_ for _ in id_features[i]]
    tmp.append(str(id_labels[i]))
    data['data'].append(tmp)
    # print(i)


for a in data.values():
    for b in a:
        if type(b) != str:
            print(b)
            np.save('instances_train', b)


# np.savetxt('instances_train.txt', data)
# print(data)

# with open('instances_train.txt', 'r') as f:
#     print(f)
