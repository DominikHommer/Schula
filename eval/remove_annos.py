import re
import sys

def remove_annotated_lines(annot_file, plain_file, out_annot, out_plain):
    with open(annot_file, 'r', encoding='utf-8') as fa, \
         open(plain_file, 'r', encoding='utf-8') as fp, \
         open(out_annot,   'w', encoding='utf-8') as foa, \
         open(out_plain,   'w', encoding='utf-8') as fop:
    
        
        annotated = fa.read()
        transcribed = fp.read()
        
        annotated = re.sub(r'\[.*?\]|\{.*?\}|<.*?>', '', annotated)
        transcribed = re.sub(r'\[.*?\]|\{.*?\}|<.*?>', '', transcribed)
        transcribed = re.sub(r'\s*([("])\s*', r' \1', transcribed)
        transcribed = re.sub(r'\s*([).,?!])', r'\1', transcribed)

        #annotated = annotated.replace('\n', ' ')
        #annotated = re.sub(r'\s+', ' ', annotated).strip()

        #transcribed = transcribed.replace('\n', ' ')
        #transcribed = re.sub(r'\s+', ' ', transcribed).strip()

        foa.write(annotated)
        fop.write(transcribed)

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: remove_annotations.py <annotated.txt> <plain.txt> "
              "<clean_annotated.txt> <clean_plain.txt>")
        sys.exit(1)
    remove_annotated_lines(*sys.argv[1:])
    
#gt groundtruth
#model das vom htr modell

#python remove_annos.py gt_da.txt model_da.txt gt_da_without_annos.txt model_da_without_annos.txt 