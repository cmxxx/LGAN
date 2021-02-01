import json
import os
import argparse
# import numpy as np

def load_mfp_json(output_dir,file_path):
    prefix=os.path.join(output_dir,os.path.basename(file_path).replace('.jsonl',''))
    train_text_out_path=prefix+"_train_text.tsv"
    train_label_out_path=prefix+"_train_label.tsv"
    dev_text_out_path=prefix+"_dev_text.tsv"
    dev_label_out_path=prefix+"_dev_label.tsv"

    with open(file_path,'r') as f, \
        open(train_text_out_path,'w') as train_t, open(train_label_out_path,'w') as train_l, \
        open(dev_text_out_path,'w') as dev_t, open(dev_label_out_path,'w') as dev_l :
        for line in f:
            result=json.loads(line)
            label = 1 if result["label"] == "real" else 0
            text=result["article"].strip().replace('\n','')
            if result["split"] == "train":
                train_t.write(f"{text}\n")
                train_l.write(f"{label}\n")
            elif result["split"] == "val":
                dev_t.write(f"{text}\n")
                dev_l.write(f"{label}\n")
            # print(result["article"])
            # break

if __name__=="__main__":
    #path="data/fakenews_machine_data/provenance/full_k40.jsonl"
    #load_mfp_json("/data",path)
    parser = argparse.ArgumentParser()


    #"NN269_Donor",#"LIAR",#"PHEMEv5",'machine_fakenews'(fakenews_machine_data)
    parser.add_argument('--DATAPATH', type=str,default='fakenews_machine_data/provenance/full_k40.jsonl')
    args = parser.parse_args()

    load_mfp_json('./',args.DATAPATH)
