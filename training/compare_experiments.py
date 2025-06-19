import os
import pandas as pd

def compare_experiments(base_dir):
    print(f"\nVergleiche Trainingsläufe unter {base_dir}")

    experiments = []
    # Gehe durch alle Unterordner im base_dir
    for run_name in os.listdir(base_dir):
        run_path = os.path.join(base_dir, run_name)
        results_csv = os.path.join(run_path, 'results.csv')

        # Nur Runs mit results.csv anschauen
        if os.path.isfile(results_csv):
            df = pd.read_csv(results_csv)

            # Wir nehmen die letzte Zeile (das letzte Epoch-Ergebnis)
            last_epoch = df.iloc[-1]

            experiments.append({
                'name': run_name,
                'metrics': last_epoch
            })

    # Sortiere Experimente nach bester mAP50-95
    experiments.sort(key=lambda x: x['metrics'].get('metrics/mAP50-95(B)', 0), reverse=True)

    print("\n Vergleichsergebnisse (sortiert nach bester mAP50-95):\n")
    for exp in experiments:
        name = exp['name']
        mAP50 = exp['metrics'].get('metrics/mAP50(B)', 0)
        mAP5095 = exp['metrics'].get('metrics/mAP50-95(B)', 0)
        precision = exp['metrics'].get('metrics/precision(B)', 0)
        recall = exp['metrics'].get('metrics/recall(B)', 0)

        print(f"🔹 {name}:")
        print(f"    📈 mAP50:      {mAP50:.4f}")
        print(f"    📈 mAP50-95:   {mAP5095:.4f}")
        print(f"    🎯 Precision:  {precision:.4f}")
        print(f"    🎯 Recall:     {recall:.4f}")
        print("")

if __name__ == '__main__':
    compare_experiments('/path/to/runs')
