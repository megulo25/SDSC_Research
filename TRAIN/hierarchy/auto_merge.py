import json
import os

# Super Classes
next_class = 0
with open('birds.json', 'r') as f:
    h0 = json.load(f)

print('{0}'.format(h0['children'][next_class]['name']))

# Sub Classes


# Next to work on is perching birds children.
def auto_merg(super_class, list_of_sub_class, next_class):
    super_class_list_of_subs_names = []
    for i in super_class['children'][next_class]['children']:
        super_class_list_of_subs_names.append(i['name'])

    sub_class_list_names = []
    dict_ = {}
    for j,k in enumerate(list_of_sub_class):
        dict_[k['name']] = j
        sub_class_list_names.append(k['name'])

    for n,m in enumerate(super_class['children'][next_class]['children']):
        m_super_name = m['name']
        try:
            idx = dict_[m_super_name]
            sub_class = list_of_sub_class[idx]
            super_class['children'][next_class]['children'][n] = sub_class
        except:
            print('Theres no {0}'.format(m_super_name))

    return super_class

new_super_class = auto_merg(h0, list_, next_class)

with open('birds.json', 'w') as w0:
    json.dump(new_super_class, w0)

with open('birds_backup.json', 'w') as w1:
    json.dump(new_super_class, w1)

for i in list_:
    try:
        os.system('rm -r *{0}*'.format(i['name']))
    except:
        print('error')
