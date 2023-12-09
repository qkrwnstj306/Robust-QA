# Easy data augmentation techniques for text classification
# Jason Wei and Kai Zou

from eda import *
import json 

#arguments to be parsed from command line
import argparse
ap = argparse.ArgumentParser()
ap.add_argument("--input", required=True, type=str, help="input file of unaugmented data")
ap.add_argument("--output", required=False, type=str, help="output file of unaugmented data")
ap.add_argument("--num_aug", required=False, type=int, help="number of augmented sentences per original sentence")
ap.add_argument("--alpha_sr", required=False, type=float, help="percent of words in each sentence to be replaced by synonyms")
ap.add_argument("--alpha_ri", required=False, type=float, help="percent of words in each sentence to be inserted")
ap.add_argument("--alpha_rs", required=False, type=float, help="percent of words in each sentence to be swapped")
ap.add_argument("--alpha_rd", required=False, type=float, help="percent of words in each sentence to be deleted")
args = ap.parse_args()

#the output file
output = None
if args.output:
    output = args.output
else:
    from os.path import dirname, basename, join
    output = join(dirname(args.input), 'eda_' + basename(args.input))

#number of augmented sentences to generate per original sentence
num_aug = 9 #default
if args.num_aug:
    num_aug = args.num_aug

#how much to replace each word by synonyms
alpha_sr = 0.1#default
if args.alpha_sr is not None:
    alpha_sr = args.alpha_sr

#how much to insert new words that are synonyms
alpha_ri = 0.1#default
if args.alpha_ri is not None:
    alpha_ri = args.alpha_ri

#how much to swap words
alpha_rs = 0.1#default
if args.alpha_rs is not None:
    alpha_rs = args.alpha_rs

#how much to delete words
alpha_rd = 0.1#default
if args.alpha_rd is not None:
    alpha_rd = args.alpha_rd

if alpha_sr == alpha_ri == alpha_rs == alpha_rd == 0:
     ap.error('At least one alpha should be greater than zero')

def valid(aug_sentence, text):
    try:
        index = aug_sentence.index(text)
        return index
    except:
        return -1
    
#generate more data with standard augmentation
def gen_eda(train_orig, output_file, alpha_sr, alpha_ri, alpha_rs, alpha_rd, num_aug=9):

    writer = open(output_file, 'w')
    custom_dict = {}
    
    with open(args.input,'r') as f:
        json_data = json.load(f)
        custom_data = []

        for data_corpus in json_data["data"]:
            custom_paragraphs = []
            for data in data_corpus["paragraphs"]:
                custom_qas = []
                #context, qas
                sentence = data["context"]
                aug_sentence = eda(sentence, alpha_sr=alpha_sr, alpha_ri=alpha_ri, alpha_rs=alpha_rs, p_rd=alpha_rd, num_aug=num_aug)
                valid_value = -1 # nat_questions 는 qas 가 없는 것도 있어서 valid_value 를 미리 -1 로 setting.
                
                for qas in data["qas"]:
                    custom_answers = []
                    for answers in qas["answers"]:
                        valid_value = valid(aug_sentence[0], answers["text"])
                        custom_answers.append({"answer_start":valid_value, "text":answers["text"]})
                        if valid_value == -1:
                            #바로 빠져나가서 valid_value -1 로 전달하기. append 안 하려고
                            break
                    custom_qas.append({"answers":custom_answers, "id": qas["id"], "question": qas["question"]})

                custom_paragraphs.append({"context":aug_sentence[0],"qas":custom_qas})

            if valid_value != -1:
                custom_data.append({"title":data_corpus["title"],"paragraphs":custom_paragraphs})
        custom_dict["data"] = custom_data
    writer.write(json.dumps(custom_dict))
    writer.close()
    print("generated augmented sentences with eda for " + train_orig + " to " + output_file + " with num_aug=" + str(num_aug))

#main function
if __name__ == "__main__":
    #generate augmented sentences and output into a new file
    gen_eda(args.input, output, alpha_sr=alpha_sr, alpha_ri=alpha_ri, alpha_rs=alpha_rs, alpha_rd=alpha_rd, num_aug=num_aug)