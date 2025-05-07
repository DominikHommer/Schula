import re
import sys

def remove_annotated_lines(annot_file, plain_file, out_annot, out_plain):
    pattern = re.compile(r"<[^>]*>|\{[^}]*\}")
    
    with open(annot_file, 'r', encoding='utf-8') as fa, \
         open(plain_file, 'r', encoding='utf-8') as fp, \
         open(out_annot,   'w', encoding='utf-8') as foa, \
         open(out_plain,   'w', encoding='utf-8') as fop:
        
        annot_lines = fa.readlines()
        plain_lines = fp.readlines()
        n = min(len(annot_lines), len(plain_lines))
        
        for i in range(n):
            if pattern.search(annot_lines[i]):
                continue
            foa.write(annot_lines[i])
            fop.write(plain_lines[i])

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: remove_annotations.py <annotated.txt> <plain.txt> "
              "<clean_annotated.txt> <clean_plain.txt>")
        sys.exit(1)
    remove_annotated_lines(*sys.argv[1:])
    
#gt groundtruth
#model das vom htr modell

#python remove_annos.py gt_da.txt model_da.txt gt_da_without_annos.txt model_da_without_annos.txt 