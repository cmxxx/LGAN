import numpy as np
from sklearn.metrics import classification_report

def get_Discriminator_where_report(d_where_label_tuple):
    d_w_report = classification_report(
        d_where_label_tuple[0],d_where_label_tuple[1],
        digits=6,
    ) 
    d_w_report_dict = classification_report(
        d_where_label_tuple[0],d_where_label_tuple[1],
        digits=6,output_dict=True
    )
    return d_w_report,d_w_report_dict


def get_D_classifier_report(d_classify_label_tuple):
    '''
    input (label, pred_label)
    '''
    d_c_report = classification_report(
        d_classify_label_tuple[0],d_classify_label_tuple[1],
        digits=6,
    )
    d_c_report_dict = classification_report(
        d_classify_label_tuple[0],d_classify_label_tuple[1],
        digits=6,
        output_dict=True
    )
    return d_c_report,d_c_report_dict

