#%%
import os
#%%
def save_dict_to_file(dic,dict_dir):
    f = open(dict_dir + '.txt', 'w')
    f.write(str(dic))
    f.close()
#%%
def generate_save_dir(atmp_dir, hyper_params):
    atmp_dir = atmp_dir + r'\atmp'

    i = 1
    tmp = True
    while tmp :
        if os.path.isdir(atmp_dir + r'\\' + str(i)) : 
            i+= 1
        else: 
            os.makedirs(atmp_dir + r'\\' + str(i))
            print("Generated atmp" + str(i))
            tmp = False
    atmp_dir = atmp_dir + r'\\' + str(i)

    save_dict_to_file(hyper_params, atmp_dir + r'\hyper_params')
    os.makedirs(atmp_dir + r'\rpn_weights')
    os.makedirs(atmp_dir + r'\dtn_weights')

    return atmp_dir