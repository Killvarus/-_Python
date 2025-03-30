from sklearn.model_selection import cross_val_score
import inspect

# Получаем исходный код функции
source_code = inspect.getsource(cross_val_score)
print(source_code)