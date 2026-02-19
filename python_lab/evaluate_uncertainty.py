import numpy as np
import matplotlib.pyplot as plt

def plot_rejection_curve(labels, probs_mc, metric_values, save_path):
    """
    metric_values: значения энтропии или MI для каждого сэмпла.
    """
    # Сортируем по убыванию неопределенности (сначала самые плохие)
    indices = np.argsort(metric_values)[::-1]
    sorted_labels = labels[indices]
    sorted_preds = probs_mc.mean(0).argmax(1)[indices]
    
    correct = (sorted_labels == sorted_preds)
    rejection_rates = np.linspace(0, 0.95, 50)
    acc_scores = []
    
    for rate in rejection_rates:
        # Отсекаем % худших
        keep_idx = int(len(correct) * rate)
        remaining_acc = correct[keep_idx:].mean()
        acc_scores.append(remaining_acc)
        
    plt.figure(figsize=(8, 5))
    plt.plot(rejection_rates * 100, acc_scores)
    plt.xlabel('Rejection Rate (%)')
    plt.ylabel('Accuracy on Remaining')
    plt.title('Rejection Curve (Uncertainty-based Filter)')
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
