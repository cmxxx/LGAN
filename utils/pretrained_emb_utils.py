import numpy as np
import pickle

def load_golve_emb(args,processor):
    glove_file = 'glove.twitter.27B.' + str(args.GLOVE_DIM) + 'd.txt'
    glove_dir = args.DATA_DIR+'/pre_embeddings/'
    emb_dict = {}
    glove = open(glove_dir + glove_file)
    for line in glove:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], dtype='float32')
        emb_dict[word] = vector
    glove.close()
    # emb_matrix = np.zeros((args.NB_WORDS, args.GLOVE_DIM))
    emb_matrix = np.random.rand(args.NB_WORDS, args.GLOVE_DIM)
    for w, i in processor._tokenizer.word_index.items():
        # The word_index contains a token for all words of the training data so we need to limit that
        if i < args.NB_WORDS:
            vect = emb_dict.get(w)
            # Check if the word from the training data occurs in the GloVe word embeddings
            # Otherwise the vector is kept with only zeros
            if vect is not None:
                emb_matrix[i] = vect
        else:
            break
    with open("{}/pretrained_emb_dim_{}_vocab_{}.pkl".format(args.OUTPUT_DIR,args.GLOVE_DIM,args.NB_WORDS),"wb") as f:
        pickle.dump(emb_matrix,f)
    return emb_matrix

def load_saved_emb(args):
    # load function
    with open("{}/pretrained_emb_dim_{}_vocab_{}.pkl".format(args.OUTPUT_DIR,args.GLOVE_DIM,args.NB_WORDS),"rb") as f:
        emb_matrix=pickle.load(f)
    return emb_matrix