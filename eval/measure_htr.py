import pandas as pd

def levenshtein(a, b):
    m, n = len(a), len(b)
    D = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1): D[i][0] = i
    for j in range(n + 1): D[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if a[i-1] == b[j-1] else 1
            D[i][j] = min(
                D[i-1][j] + 1,
                D[i][j-1] + 1,
                D[i-1][j-1] + cost
            )
    return D[m][n]

def cer(gt, pred):
    return levenshtein(list(gt), list(pred)) / max(1, len(gt))

def wer(gt, pred):
    return levenshtein(gt.split(), pred.split()) / max(1, len(gt.split()))

def read_lines(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [line.rstrip() for line in f]

gt_file   = 'gt_da_without_annos.txt'
pr_file   = 'model_da_without_annos.txt'

ground_truth = read_lines(gt_file)
predictions  = read_lines(pr_file)

n = min(len(ground_truth), len(predictions))
ground_truth = ground_truth[:n]
predictions  = predictions[:n]

for idx, (gt, pr) in enumerate(zip(ground_truth, predictions), start=1):
    line_cer = cer(gt, pr) * 100
    line_wer = wer(gt, pr) * 100
    print(f"Zeile {idx:>3}: CER = {line_cer:6.2f}%, WER = {line_wer:6.2f}%")

overall_cer = sum(cer(gt, pr) for gt, pr in zip(ground_truth, predictions)) / n
overall_wer = sum(wer(gt, pr) for gt, pr in zip(ground_truth, predictions)) / n

print("\n--- Gesamtergebnis ---")
print(f"Durchschnittliche CER: {overall_cer*100:.2f}%")
print(f"Durchschnittliche WER: {overall_wer*100:.2f}%")

rows = [
    {
        'Zeile':        i+1,
        'Ground Truth': gt,
        'Prediction':   pr,
        'CER (%)':      cer(gt, pr)*100,
        'WER (%)':      wer(gt, pr)*100,
    }
    for i, (gt, pr) in enumerate(zip(ground_truth, predictions))
]
df = pd.DataFrame(rows)

