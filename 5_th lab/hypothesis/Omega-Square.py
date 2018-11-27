import numpy
from termcolor import colored
from scipy.special import gamma

def calculate_a1(lim, S):
  def Ivz(lim, v, z):
 		return sum([((z/2.0)**(v+2.0*i))/(gamma(i+1.0)*gamma(i+v+1.0)) for i in range(lim)])

  return sum([( gamma(i + 0.5) * numpy.sqrt(4.0*i + 1.0) * scipy.exp( -( (4*i+1.0)**2 ) / (16.0*S) ) )/(  gamma(0.5) * gamma(i+1.0) ) * ( Ivz(lim, -0.25, ((4.0*i + 1.0)**2)/(16.0*S) ) - (Ivz(lim, 0.25, ((4*i + 1.0)**2)/(16.0*S)))) for i in range(lim)]) / scipy.sqrt(2 * S)

def w2_kms(self):
  """"Проверка гипотезы о распределении по критерию w2-Крамера-Мизеса-Смирнова"""
  print("\tw2-Крамера-Мизеса-Смирнова: ", end='')
  self.sequence.sort()
  n = len(self.sequence)
  s_star = 1 / (12 * n)
  for i, x in zip(range(1, n), self.sequence):
  	s_star += (self.cdf(self, x) - (2 * i - 1) / (2 * n)) ** 2

  P = 1 - self.calculate_a1(30, s_star)
  print("S* = " + str(round(s_star, 3)), end='')
  print(", значение P{S* > S} = " + str(round(P, 3)) + ", уровень значимости = 0.05 => гипотеза по критерию w2-Крамера-Мизеса-Смирнова ", end='')
  if P > 0.05:
  	print(colored('не отвергается','green'))
  else:
  	print(colored('отвергается','red'))