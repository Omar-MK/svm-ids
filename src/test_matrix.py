import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
data = {'y_predicted' : [1, 1, 3, 4, 5, 6, 4, 5, 4, 3, 2],
        'y_actual' : [1, 0, 1, 2, 4, 5, 6, 4, 5, 4, 2]}

df = pd.DataFrame(data, columns=['y_actual', 'y_predicted'])
confusion_matrix = pd.crosstab(df['y_actual'], df['y_predicted'], rownames=['Actual'], colnames=['Predicted'], margins=True)

sn.heatmap(confusion_matrix, annot=True)
plt.tight_layout()
plt.show()
