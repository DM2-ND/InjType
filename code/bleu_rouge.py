import os
import texar.torch as tx
from rouge import FilesRouge
files_rouge = FilesRouge()

def get_all_files(path):
    if os.path.isfile(path): 
        return [path]
    return os.listdir(path)


def replace_type_by_mention(raw_file_path, type_file_path):
    
    raw_file = open(raw_file_path, 'r')
    type_file = open(type_file_path, 'r')

    def get_hypothesis_list(type_file, raw_file):

        raw_file = raw_file.readlines()
        type_file = type_file.readlines()

        hypothesis_list = []
        for types, texts in zip(type_file, raw_file):
            outs = []
            ents = [ent for ent in types.strip().split()]
            for word in texts.strip().split():
                if word == '<ss>':
                    if ents:
                        outs.append(ents[0])
                        ents = ents[1:]
                    else:
                        continue
                else:
                    outs.append(word)
            hypothesis_list.append(' '.join(outs))
        return hypothesis_list

    def write_hypothesis_list(out_file_path):
        out_file = open(out_file_path, 'w')
        for hyp in hypothesis_list:
            out_file.write(f'{hyp}\n')
        out_file.close()

    hypothesis_list = get_hypothesis_list(type_file, raw_file)
    write_hypothesis_list(raw_file_path)

    raw_file.close()
    type_file.close()


def eval_bleu_rouge(filepath, typepath, refpath):

    replace_type_by_mention(filepath, typepath)

    bleu = tx.evals.file_bleu(hyp_filename=filepath, ref_filename=refpath, case_sensitive=False)
    rouge = files_rouge.get_scores(hyp_path=filepath, ref_path=refpath, avg=True)

    metrics = {'bleu': round(bleu/100, 4), 
               'rouge2': round(rouge['rouge-2']['f'], 4),
               'rougeL': round(rouge['rouge-l']['f'], 4),
            }

    return metrics
