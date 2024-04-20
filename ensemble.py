import os

import numpy as np
import pandas as pd



def ensemble(ensemble_list, save_name = 'ensb.csv'):
    prj_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(prj_dir, 'output')
    save_path = os.path.join(output_path, save_name)
    output_list = []

    for data in ensemble_list:
        data_path = os.path.join(output_path, data)
        df = pd.read_csv(data_path)
        output_list.append(np.array(df))

    ensb_result = []
    for i in range(len(output_list[0])):
        average = sum([output[i][1] / len(output_list) for output in output_list])
        ensb_result.append(average)

    ensb_dataframe = pd.DataFrame({"id": output_list[0][:, 0], "prediction": ensb_result})
    ensb_dataframe.to_csv(save_path, index=False)

if __name__ == "__main__":
    ensb_list = ['deberta_checkpoint-85293.csv',
                'roberta_checkpoint-28432.csv',
                'base_down_data_2_ep3.csv']

    ensemble(ensb_list, 'ensb_.csv')
