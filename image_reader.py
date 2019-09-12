import cv2
import numpy as np
import os
from tqdm import tqdm

file_path = '/home/armen/Desktop/Vid2Vid/nikol_dataset'


def generate_batch(file_path, batch_size):
    data_len = len(os.listdir(file_path + '/final_input'))
    for _ in range(data_len // batch_size):
        random_indices = np.random.choice(data_len, batch_size, replace=False) + 1
        target_array = []
        input_array = []
        for index in random_indices:
            target_image = cv2.imread(os.path.join(file_path, 'final_target', 'final_target{}.jpg'.format(str(index))))
            input_image = cv2.imread(os.path.join(file_path, 'final_input', 'final_input{}.jpg'.format(str(index))))
            new_target = cv2.cvtColor(target_image, cv2.COLOR_BGR2RGB)
            new_input = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
            normalized_input = new_input / 127.5 - 1
            normalized_target = new_target / 127.5 - 1
            target_array.append(normalized_target)
            input_array.append(normalized_input)
        # target_batch = np.array(target_array)
        # input_batch = np.array(input_array)
        yield input_array, target_array


# for trg_batch, inp_batch in generate_batch(file_path, 8):
#     print(trg_batch.shape)


