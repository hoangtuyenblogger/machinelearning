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


def Spearman(a,b):
  n = len(a)
  rank_a = 0
  rank_b = 0
  d = 0 ## d = rgA - rgB
  d2 = 0
  for i in range(n):
    rank_a = a[i]
    rank_b = b[i]
    d += rank_a-rank_b
    d2+= d * d
  return (6*d2)/(n*n*n - n) ## 6 * d2 / n(n*n - 1)



x=[1,2,3,4,5,6,7,8]
y=[1,3,5,4,5,7,8,4]
print('Pearson = ',Pearson(x, y))
print('Spearman = ',Spearman(x,y))