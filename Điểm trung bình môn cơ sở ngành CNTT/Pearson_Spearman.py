import math

def avg(data):
  return sum(data)/len(data)
  
def Pearson(a,b):
  avg_a = avg(a)
  avg_b = avg(b)
  
  diff_a = 0 ## độ lệch của a
  diff_b = 0 ## độ lệch của b
  diff_ab =0 ## tích độ lệch a và b
  
  diff_a2 = 0 ## bình phương độ lệch của a
  diff_b2 = 0 ## bình phương độ lệch của b
  
  n = len(a)
  ##assert n > 0
  for i in range(n):
    diff_a+= i - avg_a
    diff_b+= i - avg_b
    diff_ab += diff_a * diff_b
    
    diff_a2+= diff_a * diff_a
    diff_b2+= diff_b * diff_b
    
  return diff_ab / math.sqrt(diff_a2 * diff_b2)


def Spearman(x, y):
  dsquare = 0
  for i in range(len(x)):
    if (x[i] >= y[i]):
      dsquare += (x[i] - y[i]) ** 2
    else:
      dsquare += (y[i] - x[i]) ** 2
  return float(1 - ((6 * dsquare) / (len(x) * (len(x) ** 2 - 1))))



x=[1,2,3,4,5,6,7,8]
y=[1,3,5,4,5,7,8,4]
print('Pearson = ',Pearson(x, y))
print('Spearman = ',Spearman(x,y))